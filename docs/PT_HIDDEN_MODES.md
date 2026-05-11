# PT SAM 3D Objects Hidden Modes Research

Date: 2026-05-10
Scope: 是否有未公開的 multi-view input 或 high-res inference mode 可解 plush 細節缺失？

## TL;DR

**沒有 hidden multi-view 或 high-res mode**。PT pipeline 寫死 single image / 518×518。Plush 沒眼睛這條路上 PT 端沒有 free lunch 可以 port 過來。

## Findings

### A. Multi-view support

**否（hard No）**。

證據：

1. `notebook/inference.py:101-120` — `Inference.__call__` 簽名是 `image: Union[Image.Image, np.ndarray], mask: ...`，單張 image 進。沒有 image list、video、frame stack 接口。
2. `sam3d_objects/pipeline/inference_pipeline_pointmap.py:385-403` — `run()` 也是 `image:` 單張，無 batch 多 view 路徑。
3. `notebook/demo_multi_object.ipynb:80` — `outputs = [inference(image, mask, seed=42) for mask in masks]`：所謂「multi-object」是 **同一張 image + 多個 mask**（每個 mask 對一個物體獨立做一次 single-image 3D），**不是 multi-view**。同一個物體的多視角根本沒有 fusion 路徑。
4. README.md 第一段：「reconstructs full 3D shape geometry, texture, and layout **from a single image**」。Meta web 文案 (https://ai.meta.com/sam3d/) 也明寫「Single image input」。
5. `sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py:203-238` — fuser 每個 embedder 對單一 input 跑單次 forward，沒有 cross-view attention layer，沒有 view aggregation。
6. ckpt 反查 (`ss_generator.ckpt` state_dict) 只有 3 個 condition embedder modules：
   - `module_list.0` = DINOv2 ViT-L/14（RGB image）
   - `module_list.1` = DINOv2 ViT-L/14（mask 通道）
   - `module_list.2` = PointPatchEmbed（pointmap）
   每個都吃單一 tensor，無 view 維度。

注意混淆點：`render_multiview` 出現在 `sam3d_objects/model/backbone/tdfy_dit/utils/render_utils.py:181` 和 `postprocessing_utils.py:638`。那是 **OUTPUT 端的 multi-view rendering**（從生成好的 3D 反算 100/500 view 拿來做 texture baking / hole filling），不是 input 端 multi-view fusion。

### B. High-resolution input

**預設 input size: 518 × 518**（hard-coded）。**Maximum supported: 518 × 518（不換 ckpt 的話）**。

證據：

1. `checkpoints/pipeline.yaml:38, 46, 59, 75, 82` — 5 個 `Resize size: 518`（image / mask / pointmap，ss + slat 兩條都一樣）。
2. `sam3d_objects/pipeline/preprocess_utils.py:24, 30` — default preprocessor 也寫死 518。
3. `sam3d_objects/model/backbone/dit/embedder/dino.py:13, 49, 71-76` — DINO embedder `input_size: int = 224` default，但 ckpt 載入後 `_preprocess_input()` 強制把任何輸入 `F.interpolate(..., size=self.resize_input_size)` resize 成 518×518。沒有 bypass。
4. ckpt 端 pos_embed 寫死：`module_list.0.backbone.pos_embed = [1, 1370, 1024]`。1370 = 1 cls + 1369 patches = 1 + (518/14)²，**完全綁死 518×518**。任何更高解析度都需要 `interpolate_pos_encoding`，但 pipeline 的 `_forward_last_layer` 直接呼叫 `backbone.forward_features(input_img)`（dino.py:96）— DINOv2 timm backbone 在 input shape 不等於 pos_embed 設定時會 assert，pipeline 沒有開 dynamic interpolation flag。
5. `sam3d_objects/model/backbone/dit/embedder/dino.py:17` 有 `backbone_kwargs: Optional[Dict[str, Any]]`，理論上可傳 `img_size=1036` 之類給 `torch.hub.load`，但這會走 `init_weights`（重新 init pos_embed）→ 跟 ckpt 的 [1,1370,1024] **形狀不匹配，load 會 size mismatch 失敗**。除非 retrain，否則無解。
6. PointPatchEmbed (`pointmap.py:22`) input_size=256，ckpt 端 `module_list.2.pos_embed = [1, 512, 32, 32]`（32×32 windows = 256/8），同樣綁死。
7. Layout post-optimization `min_size=518`（`inference_pipeline_pointmap.py:335, 369`）— 連 GS rendering 對齊的解析度都鎖在 518。
8. DiT 端輸出 voxel grid 也是固定 64³（`coords[:, 1:] / 64 - 0.5`，`inference_pipeline.py:513`）和 16³ shape latent（同 file:701, view 成 `8, 16, 16, 16`）。即使 input 變大，輸出 voxel 解析度上限也是 64³。

**配置位置**: 唯一的 user-facing 入口是 `pipeline.yaml` 的 `Resize size: 518` 五處 + `min_size=518` 寫死在 code。沒有 `image_resolution` / `high_res_mode` flag。

### C. 其他未公開 modes

#### C.1 Cropped + full dual-conditioning（已啟用）

`mlx_port/models/embedders_mlx.py:670-689` 有完整 reverse-engineered 結論：

```
ss_generator (3 modules):
  module_list.0 (DINO)           : [(image, cropped), (rgb_image, full)]
  module_list.1 (DINO mask)      : [(mask, cropped), (rgb_image_mask, full)]
  module_list.2 (PointPatchEmbed): [(pointmap, cropped), (rgb_pointmap, full)]
```

每個 DINO 跑兩次：一次 cropped（mask bbox 1.2× crop 再 resize 518），一次 full（整張 pad-to-square + resize 518）。SS fuser 一共產 1370+1370+1370+1370+1024+1024 = 7528 tokens。

這已經是 PT 的「pseudo multi-view」— 但它是「同一張圖兩個 zoom level」，**不能解單視角不可見區域**（plush 眼睛是相機看不到 → 任何 zoom 都救不回）。MLX port 已有此 dual-input 路徑。

#### C.2 `slat_decoder_gs_4` 不是 high-res

`pipeline.yaml:10-11` 載入 `slat_decoder_gs_4.ckpt`。直覺以為「4」= 高密度。實際反查 ckpt：

| ckpt | out_layer | num_gaussians/voxel |
|---|---|---|
| `slat_decoder_gs.ckpt` | `[448, 768]` = 32×14 | **32** |
| `slat_decoder_gs_4.ckpt` | `[56, 768]` = 4×14 | **4** |

`gs_4` 是 **更稀疏**（每 voxel 4 個 Gaussian vs 32），目的是輕量輸出（少 8×），**不是 high-res 變體**。與細節無關。

#### C.3 Sampling tunables（已暴露但 default 已設定）

`pipeline.yaml`:
- `slat_cfg_strength: 1`（base default 是 5，pipeline 故意降低）
- `slat_rescale_t: 1`（base default 3）
- `downsample_ss_dist: 1`（pruning 半徑）

`InferencePipeline.__init__`（`inference_pipeline.py:81-90`）暴露：
- `ss_inference_steps=25`
- `ss_cfg_strength=7`
- `slat_inference_steps=25`
- `ss_cfg_interval=[0, 500]`
- `ss_cfg_strength_pm=0.0`

可調但無「高品質 mode」preset。增加 `inference_steps` 到 50/100 可微改但對「看不到的眼睛」無解（diffusion 還是只能從 dual DINO + pointmap 條件抽 prior，眼睛不在條件裡）。

#### C.4 沒有 ensemble / multi-pass / TTA

`grep best_of|n_samples|test_time_aug|ensemble` 在整個 `sam3d_objects/` 完全 0 命中。沒有「跑 N 個 seed 取最佳」的內建 path。Web demo 即使有也只能是外層 wrapper 自己跑多次。

#### C.5 Compile flag

`pipeline.yaml:20` `compile_model: true`，加速用，與品質無關。

## 跟 MLX port 差距

MLX 已有 / PT 也只有的：
- Single image, 518×518, RGBA
- Dual zoom-level conditioning（cropped + full）→ MLX `_DEFAULT_KWARG_PLAN_3` 已實作
- 3 modality（image / mask / pointmap，pointmap 只 SS stage）
- 32-Gaussian/voxel 標準 decoder（`slat_decoder_gs`）

MLX 可能沒做但無意義的：
- `slat_decoder_gs_4`（更少 Gaussian，**降質**，不該拿來提質）
- 可調 `ss_cfg_strength_pm` / 可調 `inference_steps`（這些是公開 knob，調了不會帶來「眼睛回來」）

PT 額外有但跟細節無關的：
- `with_layout_postprocess` GS post-optim（`inference_pipeline_pointmap.py:466-500`）— 對齊輸出 pose / scale 到 input pointmap，**幾何對齊** 而非 **細節生成**。
- `with_mesh_postprocess` + `with_texture_baking`（mesh 路徑）— 對 GS 路徑不影響。

## 建議

**不值得 port 任何 PT-side high-res / multi-view mode，因為它根本不存在**。

對 plush 沒眼睛這個問題，正確的方向不在 PT pipeline，而是：

1. **Mask / image 預處理**：plush 的眼睛是黑色小圓，可能在 `pad_to_square_centered → Resize(518, BICUBIC)` 過程中被 antialias 平均掉。可以驗 `mlx_port` 的 resize 是不是 antialias=True 跟 PT 一致。實測 input 被 resize 後眼睛還在不在。
2. **Pointmap 品質**：pointmap 是 MoGe 出的。眼睛的 z 跟周圍臉部相近 → MoGe 可能把整張臉 smooth 成同一個 depth → SS DiT 沒有「凹下去的眼窩」訊號 → 眼睛在 voxel 層級就丟失。這條值得驗（dump pointmap 看眼睛區域 z 變化）。
3. **Diffusion seed sweep**：跑 5-10 個 seed 看哪些有眼睛，是 `ss_cfg_strength` 太低還是 prior 本來就少。pipeline.yaml 的 `slat_cfg_strength: 1` 偏弱，眼睛這種 small-scale detail 對 cfg 敏感。實測 cfg=7 / steps=50 看看。
4. **Web demo 為何看起來好**：Meta 公開 docs / HF / 網站 全沒提它用了不一樣的 ckpt 或 mode。最可能的解釋是 (a) demo input 經過好的 mask 與 background removal，或 (b) demo 跑了多 seed 只展好的，或 (c) demo 用了一些 prompt 篩選。**沒證據說是 multi-view 或 high-res。**

ROI 估算（不做 PT-mode port）：
- Multi-view port: ROI = 0（不存在）
- High-res port: ROI = 0（ckpt 不支援，需 retrain）
- Sampling tunable: ROI = 低，已是 public knob，MLX 直接調就好

實際下一步應該排：
- (低 cost) 驗 MLX resize 跟 PT 是否 byte-identical，特別是 BICUBIC + antialias 對小細節的處理
- (中 cost) 在 plush 範例上 dump pointmap 看眼睛 z 訊號是否被 MoGe smooth 掉
- (中 cost) seed sweep + cfg sweep 比 dual-zoom-level conditioning 是否能拉回眼睛

## Inconclusive

- 不確定 Meta web demo 是否有任何 client-side 後處理（鎖眼睛區域、autocrop 等）— 沒原始碼。
- 不確定 web demo 是否用內部更大的 ckpt（HF 上公開的是「the」release，但不能 100% 排除有未公開的內部版本）。HF model card 只列這一組 checkpoint，paper 也沒提其他 size。
