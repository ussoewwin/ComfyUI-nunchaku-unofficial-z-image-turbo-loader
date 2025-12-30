# Nunchaku SDXL LoRA Loader 完全技術解説書

## 目次

1. [開発背景と動機](#開発背景と動機)
2. [技術的課題の全体像](#技術的課題の全体像)
3. [Phase 1: 基本的なLoRA適用の実装](#phase-1-基本的なlora適用の実装)
4. [Phase 2: SVDQ量子化UNetへの対応](#phase-2-svdq量子化unetへの対応)
5. [Phase 3: キーマッピングの問題解決](#phase-3-キーマッピングの問題解決)
6. [Phase 4: Q/K/V → QKV融合の実装](#phase-4-qkv--qkv融合の実装)
7. [Phase 5: パフォーマンス最適化](#phase-5-パフォーマンス最適化)
8. [Phase 6: ModelPatcher.clone()でのLoRAリーク問題](#phase-6-modelpatchercloneでのloraリーク問題)
9. [最終的な実装の詳細](#最終的な実装の詳細)
10. [まとめ](#まとめ)

---

## 開発背景と動機

### なぜ独自のLoRA Loaderが必要だったか

ComfyUIの標準LoRA Loader（`comfy.sd.load_lora_for_models`）は、通常のfp16/bf16 UNetに対しては完璧に動作する。しかし、Nunchaku SDXLのように**SVDQ（Scalable Vectorized Dynamic Quantization）で量子化されたUNet**に対しては、以下の根本的な問題があった：

1. **標準パッチングが機能しない**: ComfyUIのLoRA適用は`*.weight`パラメータを直接書き換える仕組みだが、SVDQ量子化レイヤー（`SVDQW4A4Linear`）は`qweight`、`wscales`、`proj_down`、`proj_up`などの特殊なパラメータ構造を持っており、標準的な`*.weight`が存在しない。

2. **キーマッピングの不一致**: 
   - 多くのSDXL LoRAはA1111形式（`lora_unet_input_blocks_*`）で保存されている
   - Nunchaku SDXL UNetはdiffusers形式（`down_blocks.*`）の構造
   - ComfyUIの`comfy.utils.unet_to_diffusers()`によるマッピングが、Nunchakuのdiffusers構造UNetに対して正しく機能しない場合がある

3. **アーキテクチャの違い**:
   - Nunchaku SDXLの`attn1`はQ/K/Vを融合した`to_qkv`を使用
   - 多くのLoRAは`to_q`、`to_k`、`to_v`として訓練されている
   - 融合が必要

これらの課題を解決するため、**独自のLoRA Loader実装**が必要となった。

---

## 技術的課題の全体像

### 課題1: SVDQ量子化UNetへのLoRA適用

**問題**: `SVDQW4A4Linear`には通常の`weight`パラメータが存在しない。

**解決策**: Runtime forward-add（forward時に動的にLoRAデルタを加算）

```python
# 標準的なアプローチ（機能しない）
model.layer.weight = model.layer.weight + lora_delta  # weightが存在しない

# Runtime forward-addアプローチ
def forward_with_lora(x):
    base_out = original_forward(x)
    lora_delta = compute_lora_delta(x)
    return base_out + lora_delta
```

### 課題2: キーマッピングの不一致

**問題**: A1111形式（`lora_unet_input_blocks_*`）をdiffusers形式（`lora_unet_down_blocks_*`）に変換する必要がある。

**解決策**: `comfy.utils.unet_to_diffusers()`の逆マッピングを使用し、A1111形式のLoRAキーを変換

### 課題3: Q/K/V → QKV融合

**問題**: LoRAは`attn1_to_q`、`attn1_to_k`、`attn1_to_v`として訓練されているが、Nunchaku SDXLは`attn1_to_qkv`を使用

**解決策**: 3つのLoRA行列をブロック対角行列として結合し、1つの`to_qkv` LoRAに融合

### 課題4: パフォーマンス問題

**問題**: 
- 毎forwardでのCPU↔GPU転送
- float32固定計算（重い）
- Pythonループによる複数LoRAの逐次適用

**解決策**: 
- GPU上でLoRA行列を事前準備・キャッシュ
- 複数LoRAを1つの融合行列に結合（rank連結）
- 計算dtypeを入力に合わせて動的に選択（デフォルトはfp16/bf16）

### 課題5: ModelPatcher.clone()でのLoRAリーク

**問題**: `ModelPatcher.clone()`は同じモデルオブジェクトを共有するが、LoRA patchesはper-patcherであるべき

**解決策**: `attachments`辞書を使用して、patcherごとに独立したLoRAストレージを保持

---

## Phase 1: 基本的なLoRA適用の実装

最初の実装では、標準的なComfyUI LoRA適用メカニズムをそのまま使用しようとした。

### コード1: 基本的なラッパー（初期実装）

```python
def _apply_lora_basic(model, clip, lora_sd, strength_model, strength_clip):
    """標準的なComfyUI LoRA適用"""
    return comfy.sd.load_lora_for_models(
        model, clip, lora_sd, strength_model, strength_clip
    )
```

**問題点**: 
- Nunchaku SDXL UNetに対しては`hit_bases=0`となり、LoRAが全く適用されない
- 原因: `comfy.lora.model_lora_keys_unet()`が返す`key_map`が、Nunchakuのdiffusers構造UNetと一致しない

---

## Phase 2: SVDQ量子化UNetへの対応

### コード2: SVDQ runtime LoRAの基本構造

```python
class _NunchakuSVDQRuntime:
    """
    Per-ModelPatcher runtime storage for quantized (SVDQ) LoRA matrices.
    """
    def __init__(self, debug: bool = False):
        self.debug = bool(debug)
        # module_id -> List[(down_cpu, up_cpu, scale)]
        self.loras: dict[int, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
        # ...
```

**設計思想**:
- LoRA行列（`down`、`up`）をCPU上に保持（メモリ効率）
- `module_id`（`id(mod)`）をキーとして使用（モジュールごとのLoRA管理）
- `scale`は`strength * (alpha / rank)`で事前計算

### コード3: SVDQ forwardパッチング

```python
def _ensure_svdq_forward_patched(mod, debug: bool) -> bool:
    """
    Patch SVDQW4A4Linear.forward to add runtime LoRA deltas.
    """
    if hasattr(mod, "_nunchaku_runtime_lora_patched"):
        return False
    
    orig_forward = mod.forward
    
    def _forward_with_runtime_lora(x, output=None):
        base_out = orig_forward(x, output)
        
        # Get runtime LoRAs from current patcher
        runtime = get_runtime_from_patcher(mod)
        loras = runtime.loras.get(id(mod), None)
        if not loras:
            return base_out
        
        # Apply LoRA deltas
        add = None
        for (down, up, scale) in loras:
            d = down.to(device=x.device)  # CPU -> GPU転送（遅い！）
            u = up.to(device=x.device)
            tmp = x @ d.t() @ u.t()  # 2回のGEMM
            if scale != 1.0:
                tmp = tmp * scale
            add = tmp if add is None else (add + tmp)
        
        return base_out + add
    
    mod.forward = _forward_with_runtime_lora
    return True
```

**初期実装の問題点**:
1. **毎forwardでのCPU↔GPU転送**: `down.to(device)`と`up.to(device)`が毎回実行される
2. **Pythonループ**: 複数LoRAがある場合、ループで逐次適用
3. **float32固定**: デフォルトでfloat32計算（重い）

---

## Phase 3: キーマッピングの問題解決

### 問題: hit_bases=0

多くのSDXL LoRAはA1111形式で保存されており、以下のようなキー構造を持つ：

```
lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight
lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_up.weight
```

しかし、ComfyUIの`key_map`は以下のようなdiffusers形式を期待する：

```
lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
```

### コード4: A1111 → diffusers形式への変換

```python
def _convert_a1111_unet_lora_keys_to_comfy_diffusers(lora_sd: dict, model, debug: bool):
    """
    Convert A1111 SDXL UNet LoRA keys to ComfyUI's diffusers-mapped keys.
    """
    # Build reverse mapping using comfy.utils.unet_to_diffusers()
    unet_config = model.model.model_config.unet_config
    diff_map = comfy.utils.unet_to_diffusers(unet_config)
    
    # Reverse base mapping (strip .weight/.bias)
    reverse_base: dict[str, str] = {}
    for dkey, okey in diff_map.items():
        for suffix in (".weight", ".bias"):
            if dkey.endswith(suffix) and okey.endswith(suffix):
                reverse_base[okey[:-len(suffix)]] = dkey[:-len(suffix)]
    
    # Perform conversion
    converted = {}
    for k, v in lora_sd.items():
        base = _lora_base_key(k)
        if base.startswith("lora_unet_input_blocks_") or \
           base.startswith("lora_unet_output_blocks_") or \
           base.startswith("lora_unet_middle_block_"):
            # Convert A1111 format to dot notation
            base_wo = base[len("lora_unet_"):]
            input_dot = _a1111_base_to_input_dot(base_wo)
            if input_dot:
                diffusers_dot = reverse_base.get(input_dot)
                if diffusers_dot:
                    new_base = "lora_unet_" + diffusers_dot.replace(".", "_")
                    new_k = new_base + k[len(base):]
                    converted[new_k] = v
                    continue
        converted[k] = v
    
    return converted
```

### コード5: A1111形式のパース（重要な部分）

```python
def _a1111_tail_to_dot(tail: str) -> str:
    """
    Transform the tail part after block indices into dot-form.
    Example: transformer_blocks_0_attn1_to_q -> transformer_blocks.0.attn1.to_q
    """
    # CRITICAL: A1111 uses "_" as hierarchy separator, but some tokens include "_"
    # We must protect tokens like "to_q", "proj_in" before converting
    
    # 1) Protect tokens that must keep "_"
    out = tail
    out = out.replace("proj_in", "PROJIN")
    out = out.replace("to_q", "TOQ")
    # ... more protections ...
    
    # 2) Structural rewrites
    out = out.replace("transformer_blocks_", "transformer_blocks.")
    out = out.replace("ff_net_", "ff.net.")
    out = out.replace("attn1_to_", "attn1.")
    
    # 3) Convert remaining "_" to "."
    out = out.replace("_", ".")
    
    # 4) Restore protected tokens
    out = out.replace("PROJIN", "proj_in")
    out = out.replace("TOQ", "to_q")
    # ...
    
    return out
```

**なぜこれが重要か**: 
- 単純な`"_" -> "."`置換では、`to_q`が`to.q`になってしまい、マッピングが失敗する
- `proj_in`が`proj.in`になるのも同様

---

## Phase 4: Q/K/V → QKV融合の実装

### 問題の詳細

Nunchaku SDXLの`attn1`は、Q/K/Vを融合した`to_qkv`を使用する：

```python
# Nunchaku SDXL UNet structure
attn1.to_qkv  # (in_features, 3 * out_features)
```

しかし、多くのLoRAは以下として訓練されている：

```python
attn1_to_q.lora_down.weight  # (rank_q, in_features)
attn1_to_q.lora_up.weight    # (out_features, rank_q)
attn1_to_k.lora_down.weight  # (rank_k, in_features)
attn1_to_k.lora_up.weight    # (out_features, rank_k)
attn1_to_v.lora_down.weight  # (rank_v, in_features)
attn1_to_v.lora_up.weight    # (out_features, rank_v)
```

### コード6: Q/K/V融合の実装

```python
def _fuse_sdxl_attn1_qkv_lora(lora_sd: dict, debug: bool):
    """
    Fuse (attn1_to_q, attn1_to_k, attn1_to_v) LoRA weights into (attn1_to_qkv).
    """
    # Group by prefix before "_attn1_to_"
    groups: dict[str, dict[str, str]] = {}
    for k in lora_sd.keys():
        base = _lora_base_key(k)
        if "_attn1_to_" in base and \
           (base.endswith("_attn1_to_q") or base.endswith("_attn1_to_k") or base.endswith("_attn1_to_v")):
            prefix = base.rsplit("_attn1_to_", 1)[0]
            suffix = base.rsplit("_attn1_to_", 1)[1]  # "q", "k", or "v"
            groups.setdefault(prefix, {})[suffix] = base
    
    out = dict(lora_sd)
    
    for prefix, parts in groups.items():
        fused_base = f"{prefix}_attn1_to_qkv"
        
        # Collect present projections
        present = {}
        for p in ("q", "k", "v"):
            b = parts.get(p)
            if not b:
                continue
            up = lora_sd.get(f"{b}.lora_up.weight")
            dn = lora_sd.get(f"{b}.lora_down.weight")
            if up is None or dn is None:
                continue
            present[p] = (b, up, dn)
        
        if not present:
            continue
        
        # Determine dimensions
        any_b, any_up, any_dn = next(iter(present.values()))
        in_dim = int(any_dn.shape[1])
        dim = int(any_up.shape[0])  # out_features per projection
        
        # Build fused matrices
        ranks = {p: int(dn.shape[0]) for p, (_, _, dn) in present.items()}
        r_total = sum(ranks.values())
        
        # Fused down: (r_total, in_dim) - concatenate rows
        dn_fused = any_dn.new_zeros((r_total, in_dim))
        # Fused up: (3 * dim, r_total) - block diagonal in output dimension
        up_fused = any_up.new_zeros((3 * dim, r_total))
        
        rank_cursor = 0
        for p, row_off in (("q", 0), ("k", dim), ("v", 2 * dim)):
            if p not in present:
                continue
            b, up, dn = present[p]
            r = int(dn.shape[0])
            
            # Get alpha scale for this projection
            alpha = lora_sd.get(f"{b}.alpha", None)
            alpha_scale = (float(alpha.item()) / r) if alpha is not None else 1.0
            
            # Copy down matrix
            dn_fused[rank_cursor : rank_cursor + r, :] = dn
            # Copy up matrix with alpha scaling, into correct output segment
            up_fused[row_off : row_off + dim, rank_cursor : rank_cursor + r] = up * alpha_scale
            rank_cursor += r
        
        # Store fused LoRA
        out[f"{fused_base}.lora_down.weight"] = dn_fused
        out[f"{fused_base}.lora_up.weight"] = up_fused
        out[f"{fused_base}.alpha"] = dn_fused.new_tensor(float(r_total))
        
        # Remove original q/k/v keys
        for p in ("q", "k", "v"):
            if p not in present:
                continue
            b = present[p][0]
            for sfx in (".lora_down.weight", ".lora_up.weight", ".alpha"):
                out.pop(f"{b}{sfx}", None)
    
    return out
```

**設計のポイント**:

1. **ブロック対角行列**: `up_fused`は`(3*dim, r_total)`で、Q/K/Vの出力をそれぞれ`[0:dim]`、`[dim:2*dim]`、`[2*dim:3*dim]`に配置

2. **rank連結**: `dn_fused`は`(r_total, in_dim)`で、Q/K/Vのrankを連結（`r_total = rq + rk + rv`）

3. **alpha scalingの保持**: 各プロジェクションの`alpha / rank`を`up`行列に事前乗算して保持

4. **出力スケール**: `fused_base.alpha = r_total`とすることで、ComfyUIの`(alpha / rank) * scale`が`1.0`となり、元のスケーリングが保持される

**数学的な説明**:

元のLoRA適用（Q/K/Vそれぞれ）:
```
delta_q = (alpha_q / rank_q) * (up_q @ (x @ down_q^T))
delta_k = (alpha_k / rank_k) * (up_k @ (x @ down_k^T))
delta_v = (alpha_v / rank_v) * (up_v @ (x @ down_v^T))
```

融合後のLoRA適用（QKV）:
```
# x: (N, in_dim)
# dn_fused: (r_total, in_dim) = [dn_q; dn_k; dn_v]  (垂直連結)
# up_fused: (3*dim, r_total) = [up_q_scaled, 0, 0; 0, up_k_scaled, 0; 0, 0, up_v_scaled]  (ブロック対角)
# 
tmp = x @ dn_fused^T  # (N, r_total)
delta_qkv = up_fused^T @ tmp  # (N, 3*dim)
# delta_qkv = [delta_q; delta_k; delta_v]
```

---

## Phase 5: パフォーマンス最適化

### 問題: LoRA適用が異常に遅い

初期実装では、LoRA適用時に**fp16 UNetよりも遅くなる**という深刻な問題が発生した。

原因:
1. **毎forwardでのCPU↔GPU転送**: `down.to(device)`、`up.to(device)`
2. **float32固定計算**: `x.to(dtype=torch.float32) @ d.t()`
3. **Pythonループ**: 複数LoRAを逐次適用

### コード7: GPU上での事前準備とキャッシング

```python
class _NunchakuSVDQRuntime:
    def __init__(self, debug: bool = False):
        # ...
        # Prepared per-module fused matrices cached per-device/dtype
        # module_id -> {"device": torch.device, "dtype": torch.dtype, "down_t": Tensor, "up_t": Tensor}
        self.prepared: dict[int, dict] = {}
        # module_id set that indicates prepared cache must be rebuilt
        self.dirty: set[int] = set()
```

### コード8: 融合行列の事前準備

```python
def _forward_with_runtime_lora(x, output=None):
    base_out = orig_forward(x, output)
    
    # ... get runtime and loras ...
    
    device = base_out.device
    compute_dtype = _runtime_lora_compute_dtype(x, base_out)  # デフォルトは x.dtype
    
    # Check if prepared cache needs rebuild
    prep = runtime.prepared.get(id(mod), None)
    needs_rebuild = False
    if not isinstance(prep, dict):
        needs_rebuild = True
    elif prep.get("device") != device or prep.get("dtype") != compute_dtype:
        needs_rebuild = True
    if id(mod) in runtime.dirty:
        needs_rebuild = True
    
    if needs_rebuild:
        # Fuse multiple LoRAs into a single pair of matrices
        downs = []
        ups = []
        for (down_cpu, up_cpu, scale) in loras:
            d = down_cpu.to(device=device, dtype=compute_dtype)
            u = up_cpu.to(device=device, dtype=compute_dtype)
            if scale != 1.0:
                u = u * float(scale)  # Scaleを事前乗算
            downs.append(d)
            ups.append(u)
        
        # Concatenate to reduce kernel launches
        d_cat = torch.cat(downs, dim=0).contiguous()  # (R_total, in)
        u_cat = torch.cat(ups, dim=1).contiguous()    # (out, R_total)
        
        # Transpose for efficient matmul
        down_t = d_cat.transpose(0, 1).contiguous()   # (in, R_total)
        up_t = u_cat.transpose(0, 1).contiguous()     # (R_total, out)
        
        runtime.prepared[id(mod)] = {
            "device": device,
            "dtype": compute_dtype,
            "down_t": down_t,
            "up_t": up_t
        }
        runtime.dirty.discard(id(mod))
        prep = runtime.prepared[id(mod)]
    
    # Single 2-GEMM computation
    add = (x2 @ prep["down_t"]) @ prep["up_t"]
    
    return base_out + add.to(dtype=base_out.dtype)
```

**最適化のポイント**:

1. **GPU上で事前準備**: LoRA行列を最初のforward時にGPU上に転送し、キャッシュ

2. **複数LoRAの融合**: 複数のLoRA行列を`torch.cat`で連結し、1回の2-GEMM計算に統合
   ```
   元: for each lora: add += (x @ down_i^T) @ up_i^T  (N回のループ)
   最適化後: add = (x @ [down_1; down_2; ...]^T) @ [up_1, up_2, ...]^T  (1回の計算)
   ```

3. **計算dtypeの動的選択**: デフォルトは`x.dtype`（通常はfp16/bf16）を使用。環境変数`NUNCHAKU_SDXL_SVDQ_RUNTIME_DTYPE`で制御可能

4. **NaN/Infチェックのオプション化**: デフォルトはOFF（非常に重い）。`NUNCHAKU_SDXL_SVDQ_RUNTIME_CHECK_FINITE=1`で有効化

### コード9: 計算dtypeの決定

```python
def _runtime_lora_compute_dtype(x: torch.Tensor, base_out: torch.Tensor) -> torch.dtype:
    """
    Decide compute dtype for runtime SVDQ LoRA matmuls.
    Default is to match `x.dtype` (fast), instead of forcing float32 (slow).
    """
    mode = os.getenv("NUNCHAKU_SDXL_SVDQ_RUNTIME_DTYPE", "x").strip().lower()
    if mode in ("x", "input"):
        return x.dtype  # デフォルト: 入力のdtypeを使用
    if mode in ("out", "output"):
        return base_out.dtype
    if mode == "fp32":
        return torch.float32  # 明示的に指定した場合のみ
    # ...
```

---

## Phase 6: ModelPatcher.clone()でのLoRAリーク問題

### 問題の詳細

ComfyUIの`ModelPatcher.clone()`は、同じモデルオブジェクト（`model.model`）を共有するが、patchesは各patcherごとに独立しているべき。しかし、初期実装ではLoRA行列をモジュール自体に保存していたため、`clone()`したpatcher間でLoRAがリークしてしまう。

### コード10: attachmentsを使ったper-patcherストレージ

```python
def _apply_runtime_lora_to_svdq_modules(model, lora_converted, strength, debug):
    # ...
    # Per-patcher storage (prevents LoRA leakage across clones)
    attachments = getattr(model, "attachments", None)
    if not isinstance(attachments, dict):
        attachments = {}
        setattr(model, "attachments", attachments)
    
    runtime = attachments.get("_nunchaku_svdq_runtime", None)
    if not isinstance(runtime, _NunchakuSVDQRuntime):
        runtime = _NunchakuSVDQRuntime(debug=debug)
        attachments["_nunchaku_svdq_runtime"] = runtime
```

### コード11: forward内でのpatcher検出

```python
def _forward_with_runtime_lora(x, output=None):
    base_out = orig_forward(x, output)
    
    # Get root_model from module
    root_id = getattr(mod, "_nunchaku_runtime_root_id", None)
    root_model = _NUNCHAKU_ROOT_MODEL_WEAK.get(root_id, None)
    
    # Get current patcher (set by ComfyUI during sampling)
    patcher = getattr(root_model, "current_patcher", None)
    if patcher is None:
        patcher = getattr(root_model, "_nunchaku_runtime_last_patcher", None)
    
    # Get runtime from patcher's attachments
    attachments = getattr(patcher, "attachments", None)
    runtime = attachments.get("_nunchaku_svdq_runtime", None) if attachments else None
```

**設計のポイント**:

1. **weak reference**: `_NUNCHAKU_ROOT_MODEL_WEAK`で循環参照を回避

2. **current_patcher**: ComfyUIがサンプリング中に`model.current_patcher`を設定することを利用

3. **fallback**: `current_patcher`が設定されていない場合、`_nunchaku_runtime_last_patcher`を使用

### コード12: clone()時のストレージコピー

```python
class _NunchakuSVDQRuntime:
    def on_model_patcher_clone(self):
        n = _NunchakuSVDQRuntime(self.debug)
        n.loras = {k: v[:] for k, v in self.loras.items()}  # コピー
        # Do NOT copy prepared GPU tensors (device/offload may differ)
        n.prepared = {}
        n.dirty = set(n.loras.keys())  # 最初のforwardで再構築
        return n
```

**重要な点**: `prepared`（GPU上の融合行列）はコピーしない。理由:
- 異なるpatcherは異なるdevice/offload設定を持つ可能性がある
- 最初のforward時に自動的に再構築される（`dirty`フラグで制御）

---

## 最終的な実装の詳細

### コード13: state_dict-only key_map構築

Nunchaku SDXL UNetはdiffusers構造だが、ComfyUIの`comfy.lora.model_lora_keys_unet()`は、SDXLに対して`input_blocks`/`output_blocks`へのリマップを適用してしまう。これを回避するため、`state_dict`から直接`key_map`を構築する。

```python
def _build_unet_key_map_state_dict_only(base_model) -> dict:
    """
    Build UNet key_map ONLY from the actual state_dict keys,
    without calling comfy.utils.unet_to_diffusers().
    """
    key_map: dict[str, str] = {}
    sd = base_model.state_dict()
    for k in sd.keys():
        if not k.startswith("diffusion_model."):
            continue
        
        if k.endswith(".weight"):
            key_lora = k[len("diffusion_model.") : -len(".weight")].replace(".", "_")
            key_map[f"lora_unet_{key_lora}"] = k
            key_map[k[:-len(".weight")]] = k  # generic format
    
    return key_map
```

### コード14: SVDQキーの事前ストリッピング

SVDQ runtimeで処理されるキーを、標準LoRA loaderに渡す前にストリップすることで、「lora key not loaded」スパムを防止。

```python
def _strip_runtime_svdq_keys(lora_converted, model, debug):
    """
    Remove UNet LoRA entries that will be applied via SVDQ runtime.
    """
    # Identify bases that resolve to actual SVDQ modules
    runtime_bases: set[str] = set()
    for k in lora_converted.keys():
        base = _lora_base_key_from_any(k)
        if not base.startswith("lora_unet_"):
            continue
        mod_path = _svdq_lora_base_to_module_dot_path(base)
        if mod_path is None:
            continue
        try:
            mod = _resolve_dot_path(diffusion_model, mod_path)
            if isinstance(mod, SVDQW4A4Linear):
                runtime_bases.add(base)
        except Exception:
            continue
    
    # Filter out runtime bases
    out = {}
    for k, v in lora_converted.items():
        base = _lora_base_key_from_any(k)
        if base in runtime_bases:
            continue
        out[k] = v
    
    return out
```

### コード15: カバレッジレポート

「完全にマッピング」を保証するため、全てのUNet LoRA base keyが、standardまたはruntimeのどちらかに割り当てられているかを確認。

```python
def _coverage_report_for_unet(lora_converted, key_map, runtime_bases, runtime_skipped_bases, debug):
    """
    Verify that every UNet base key is either standard-mappable or runtime-handled.
    """
    bases: set[str] = set()
    for k in lora_converted.keys():
        base = _lora_base_key_from_any(k)
        if base.startswith("lora_unet_"):
            bases.add(base)
    
    standard_bases = {b for b in bases if b in key_map}
    runtime_hit = {b for b in bases if b in runtime_bases}
    unmapped = bases - standard_bases - runtime_hit
    
    # Report statistics
    stats = {
        "unet_bases_total": len(bases),
        "standard_bases": len(standard_bases),
        "runtime_bases": len(runtime_hit),
        "unmapped_bases": len(unmapped),
        # ...
    }
    
    if debug and stats["unmapped_bases"] > 0:
        raise ValueError(f"UNet LoRA unmapped bases detected: {unmapped}")
    
    return stats
```

---

## まとめ

### 開発過程で直面した主な課題

1. **SVDQ量子化UNetへのLoRA適用**: Runtime forward-addの実装
2. **キーマッピングの不一致**: A1111形式→diffusers形式への変換
3. **Q/K/V → QKV融合**: ブロック対角行列による融合
4. **パフォーマンス問題**: GPUキャッシング、融合行列、dtype最適化
5. **ModelPatcher.clone()でのリーク**: attachmentsによるper-patcherストレージ

### 最終的な実装の特徴

- **完全なカバレッジ**: 全てのUNet LoRA base keyがstandardまたはruntimeに割り当てられることを保証
- **高性能**: GPU上での事前準備、融合行列、動的dtype選択により、量子化UNetの速度を維持
- **堅牢性**: ModelPatcher.clone()でのLoRAリークを防止、エラーハンドリングの充実
- **可観測性**: 詳細なデバッグログ、カバレッジレポート、LoRAタイプ検出

### 今後の改善可能性

1. **DoRAサポート**: 現在は`dora_scale`などのvariant keyを検出して報告するが、実際の適用は未対応
2. **LoHa/LoKr対応**: 現在はvanilla LoRAのみサポート
3. **更なる最適化**: CUDA kernelによる融合GEMMの最適化

---

**この実装は、Nunchaku SDXLのような特殊な量子化UNetに対して、標準的なLoRAを適用可能にするための包括的なソリューションである。**

