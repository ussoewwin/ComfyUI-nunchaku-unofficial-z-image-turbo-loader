# 非公式Z-Image-Turboローダー側のControlNet対応追加・修正の詳細解説

## 追加修正が必要になった原因

ComfyUI標準のModelPatcherは、ControlNetパッチを`transformer_options["patches"]["double_block"]`に登録し、各transformerブロック後に呼び出す仕組みになっている。

QwenImage（ComfyUI-nunchaku）には最初からこの仕組みが実装されている。しかし、非公式ZITローダーはdiffusers/Nunchakuの`NunchakuZImageTransformer2DModel.forward()`をそのまま使っているため、`double_block`パッチを呼び出さず、ControlNetが効かない状態だった。

そこで、非公式ZITローダーにも同じ仕組みを追加する必要があった。

## 追加・修正したファイル

1. `ComfyUI-nunchaku-unofficial-z-image-turbo-loader/__init__.py`
2. `ComfyUI-nunchaku-unofficial-z-image-turbo-loader/model_base/zimage.py`

## 修正1: __init__.py - ModelPatcher統合用モンキーパッチ

### 修正位置
72行目から306行目

### 概要
`NunchakuZImageTransformer2DModel.forward`をモンキーパッチし、各transformerブロック後に`double_block`パッチを呼び出すようにする。

重要な点は、**モジュール階層を変更しない**こと。これにより、LoRAのモジュール名一致が保たれる。

### コード構造

```python
if not getattr(NunchakuZImageTransformer2DModel, "_comfyui_mp_patched", False):
    _orig_forward = NunchakuZImageTransformer2DModel.forward
    
    def _apply_double_block_patches(...):
        # double_blockパッチを適用する処理
    
    def _ensure_layers_patched_in_place(parent):
        # 各layer.forwardをin-placeでパッチ
    
    def _patched_forward(self, x, t, cap_feats=None, ...):
        # forwardの前にtransformer_optionsを保存・設定
    
    NunchakuZImageTransformer2DModel.forward = _patched_forward
```

### 1. _apply_double_block_patches関数（78行目から215行目）

各transformerブロックの後に呼び出される関数。

#### パラメータの説明
- `parent`: 親モデル。`transformer_options`などを保存するために使う
- `block_index`: ブロック番号
- `unified_in`: ブロック入力。`[img_tokens, txt_tokens]`の順
- `unified_out`: ブロック出力
- `adaln_input`: AdaLN入力

#### 処理フロー

**transformer_optionsからpatchesを取得する**

```python
transformer_options = getattr(parent, "_comfyui_transformer_options", None)
patches = transformer_options.get("patches", {})
double_block_patches = patches.get("double_block", [])
```

パッチがない場合は、`unified_out`をそのまま返す。

**patch_in辞書を構築する**

ControlNetパッチに渡す入力データを作成する。

```python
patch_in = {
    "x": patch_x,  # (B, C, H, W)に変換した元の入力
    "block_index": block_index,
    "transformer_options": transformer_options,
    "img": unified[:, :img_len],  # 画像トークン部分
    "txt": unified[:, img_len:img_len + cap_len],  # テキストトークン部分
    "img_input": unified_in[:, :img_len],  # ブロック前の画像トークン
    "pe": pe_cached,  # RoPE埋め込み（ComfyUI形式）
    "vec": adaln_input,  # AdaLN入力
    "block_type": "double"
}
```

**xの変換処理（120行目から125行目）**

Z-Imageは`List[torch.Tensor]`形式で、各要素の形状は`(C, F, H, W)`。一方、ZImageControlPatchは`(B, C, H, W)`形式を期待する。そのため、リストをスタックしてバッチ次元を追加し、F=1の場合は削除する。

```python
if isinstance(original_x_list, list) and len(original_x_list) > 0:
    patch_x = torch.stack(original_x_list, dim=0)  # (B, C, F, H, W)
    if patch_x.shape[2] == 1:
        patch_x = patch_x.squeeze(2)  # (B, C, H, W)
```

**RoPE埋め込み生成処理（142行目から202行目）**

ComfyUI標準の`EmbedND`を使ってRoPE埋め込みを生成する。座標IDは`(batch, seq_len, 3)`の形状で、各要素は`[cap_offset, y, x]`。`rope_options`（scale/shift）も考慮する。

```python
rope_embedder = EmbedND(dim=head_dim, theta=rope_theta, axes_dim=list(axes_dims))
ids = torch.zeros((b, img_len, 3), dtype=torch.float32, device=unified.device)
ids[:, :n, 0] = cap_offset  # cap_len + 1
ids[:, :n, 1] = ys[:n]  # y座標
ids[:, :n, 2] = xs[:n]  # x座標
pe_img = rope_embedder(ids).movedim(1, 2)
```

**パッチ適用処理（208行目から213行目）**

パッチの戻り値を`unified`テンソルに反映する。

```python
patch_out = p(patch_in)
if isinstance(patch_out, dict) and unified is not None:
    if "img" in patch_out:
        unified[:, :img_len] = patch_out["img"]
    if "txt" in patch_out:
        unified[:, img_len:img_len + cap_len] = patch_out["txt"]
```

### 2. _ensure_layers_patched_in_place関数（217行目から243行目）

各layerのforwardメソッドをin-placeでパッチする。モジュール階層は変更しない。

```python
def _ensure_layers_patched_in_place(parent: "NunchakuZImageTransformer2DModel"):
    if getattr(parent, "_comfyui_layers_patched_in_place", False):
        return
    layers = getattr(parent, "layers", None)
    for idx, layer in enumerate(layers):
        if getattr(layer, "_comfyui_double_block_patched", False):
            continue
        orig_layer_forward = layer.forward
        
        def _layer_forward_patched(*args, __orig=orig_layer_forward, __idx=idx, __parent=parent, **kwargs):
            unified_in = args[0] if len(args) > 0 else None
            adaln_input = args[3] if len(args) > 3 else None
            out = __orig(*args, **kwargs)
            return _apply_double_block_patches(__parent, __idx, unified_in, out, adaln_input)
        
        layer.forward = _layer_forward_patched
        layer._comfyui_double_block_patched = True
```

この方法により、各layerの`forward`をラップし、ブロック後に`_apply_double_block_patches`を呼び出す。モジュール階層は変更しないため、`layers.0.attention.*`のパスが維持され、LoRAが正しく一致する。

### 3. _patched_forward関数（245行目から302行目）

メインのforwardメソッドをラップする関数。

- `transformer_options`、`x`、`cap_feats`をインスタンス属性に保存する
- `img_len`と`cap_len`を計算して保存する
- 初回呼び出し時に`_ensure_layers_patched_in_place`でlayerをパッチする
- 最後に元のforwardを呼び出す

```python
def _patched_forward(self, x, t, cap_feats=None, *args, control=None, transformer_options=None, **kwargs):
    if isinstance(transformer_options, dict):
        self._comfyui_transformer_options = transformer_options
    else:
        self._comfyui_transformer_options = {}
    self._comfyui_original_x_list = x
    self._comfyui_cap_feats = cap_feats
    
    if isinstance(cap_feats, list) and len(cap_feats) > 0:
        self._comfyui_cap_len = int(cap_feats[0].shape[0])
    
    if isinstance(x, list) and len(x) > 0:
        _, f, h, w = x[0].shape
        img_tokens = (h // patch_size) * (w // patch_size) * (f // f_patch_size)
        img_tokens = int(((img_tokens + 31) // 32) * 32)  # 32の倍数に切り上げ
        self._comfyui_img_len = img_tokens
    
    _ensure_layers_patched_in_place(self)
    
    return _orig_forward(self, x, t, cap_feats=cap_feats, **call_kwargs)
```

## 修正2: model_base/zimage.py - transformer_optionsの伝達

### 修正位置
138行目から148行目

### 修正前のコード

```python
model_output = self.diffusion_model(xc_list, t_zimage, cap_feats=cap_feats, **zimage_kwargs)
```

### 修正後のコード

```python
forward_kwargs = dict(zimage_kwargs)
try:
    sig = inspect.signature(self.diffusion_model.forward)
    params = set(sig.parameters.keys())
    if "transformer_options" in params:
        forward_kwargs["transformer_options"] = transformer_options
    if "control" in params:
        forward_kwargs["control"] = control
except Exception:
    pass

model_output = self.diffusion_model(xc_list, t_zimage, cap_feats=cap_feats, **forward_kwargs)
```

### 意味の説明

`inspect.signature`を使って、`forward`メソッドが`transformer_options`や`control`パラメータを受け付けるかどうかを確認する。受け付ける場合のみ、それらを渡す。これにより、`__init__.py`の`_patched_forward`に`transformer_options`が届くようになる。

## 重要な設計判断

### モジュール階層を変更しない（LoRA互換性の確保）

最初の実装では、`layers[i]`をラッパーオブジェクトで置き換える方法を試した。しかし、LoRAが`layers.0.attention.*`というパスでモジュールを探索するため、階層が変わると一致しなくなってしまう。

最終的には、各`layer.forward`をin-placeでモンキーパッチする方式に変更した。これにより、モジュール階層が維持され、ControlNetとLoRAが両立する。

## まとめ

非公式ZITローダーにControlNet対応を追加した。`__init__.py`で`forward`と各`layer.forward`をラップし、ブロック後に`double_block`パッチを呼び出すようにした。モジュール階層は変更せず、`model_base/zimage.py`で`transformer_options`を伝達するようにした。これにより、ControlNetとLoRAの両立を実現した。

