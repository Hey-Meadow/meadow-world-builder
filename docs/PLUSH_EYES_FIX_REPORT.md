# SAM 3D Objects MLX：Plush 眼睛修復報告

Date: 2026-05-10
Hardware: Apple M1 Max (64 GB)
Status: ✅ FIXED

## TL;DR

**真正 root cause**：MLX inference pipeline 強制把 SS DiT 輸出 prune 到 16,000 voxels（為了 SLAT 記憶體），但 plush 需要 21,477 voxels（PT 跑 26,654）。**Prune 砍掉 25-40% voxels，包含臉部細節（眼睛、嘴巴）區域**。

**修法**：把 `max_voxels = 16000` 硬編碼改成可由 `MLX_SS_MAX_VOXELS` 環境變數控制。設 100,000 後不 prune，Gaussian 數從 64k → 687k，B-channel std 從 0.012 → 0.043（**3.6× saturation 提升**）。

---

## 偵錯路徑（時間軸）

| 階段 | 動作 | 結果 |
|---|---|---|
| 1 | 用戶反饋 plush 沒眼睛 | 啟動偵錯 |
| 2 | 嘗試 fp16 / scaling_clamp 等超參數調整 | 部分改善但沒解 |
| 3 | 重跑 PT vs MLX 完整 dump（修了 6-kwarg bug 後第一次）| ss_cond 0.97, voxels jaccard 0.745 ✅ |
| 4 | 餵 PT slat_final 進 MLX GS decoder | fdc cosine **0.99999** → GS decoder 沒問題 |
| 5 | 餵 MLX slat_final（no-prune, 26k voxels）進 MLX GS decoder | fdc B-std 0.59 ≈ PT 0.57 ✅ |
| 6 | **發現 16k prune 是 root cause**：no-prune slat 算對，prune 後算錯 | 鎖定 |
| 7 | 開 `MLX_SS_MAX_VOXELS=100000` 跑 plush + gs_32 | 眼睛出現 ✅ |

## 為什麼 prune 會破壞細節

Prune 用 surface heuristic（鄰居數低 → 表面 voxel）+ 隨機 subsample 砍到 16k。但：

1. PT 跑 plush **26,654 voxels** — 沒任何 prune
2. MLX SS DiT 跑 plush **21,477 voxels** — 比 PT 少 19% 但接近
3. MLX 把 21k → 16k，**砍掉 5,477 voxels (25%)**
4. 砍掉的 voxels 主要在臉部周圍（高密度、低 surface score 區）→ 眼睛位置 voxels 沒了
5. SLAT 在剩下 16k voxels 上跑，沒辦法重建眼睛
6. 雪上加霜：MLX 預設 gs_4 decoder = 4 Gaussians/voxel，PT 用 gs_32 = 32/voxel，**Gaussian 容量再砍 8×**

## 修法 diff

```python
# mlx_port/models/pipeline_mlx.py:1027
- max_voxels = 16000
+ import os as _os
+ max_voxels = int(_os.environ.get("MLX_SS_MAX_VOXELS", "16000"))
```

CLI 使用：
```bash
MLX_SS_MAX_VOXELS=100000 SLAT_GS_VARIANT=gs \
  python mlx_port/infer_mlx.py \
    --image plush.png --mask plush_mask.png \
    --use-moge --use-shortcut --slat-cfg 5 --slat-steps 25 \
    --dtype fp16 --no-prune-outliers --out plush.ply
```

---

## 速度數據（M1 Max, fp16）

### plush 完整管線（21,477 voxels, gs_32）

| 階段 | 耗時 |
|---|---:|
| preprocess | 0.03 s |
| MoGe (DINOv2 ViT-L pointmap) | 1.77 s |
| ss_embed (DINO + PointPatchEmbed) | 1.98 s |
| ss_flow (4-step shortcut) | 6.60 s |
| slat_embed | 1.78 s |
| **slat_flow (25-step CFG-5)** | **82.59 s** ← 主要耗時 |
| gs_decode (gs_32) | 1.25 s |
| **總計** | **96.0 s** |

### 對比歷史版本（plush）

| 設定 | N (Gaussians) | 時間 | B-std |
|---|---:|---:|---:|
| **v3 FINAL (no prune + gs_32 + fp16)** ✅ | **687k** | **96 s** | **0.043** |
| v2 plush_PT_match (16k + gs_32 + fp16) | 512k | 99 s | 0.012 |
| v1 chair-grade (16k + gs_4 + bf16 + shortcut) | 64k | 86 s | varies |

### chair / table（v1 基準）

| 物件 | N (Gaussians) | 時間 |
|---|---:|---:|
| chair | 64k (gs_4) | 86 s |
| table | 64k (gs_4) | 94 s |

注意：chair / table 用 gs_4 已 production-grade，沒這個 prune 問題（chair 只 15k voxels < 16k cap，根本沒被 prune）。

---

## 對齊度驗證（vs PT reference）

| 指標 | 1 天前舊報告 | 當前 |
|---|---|---|
| ss_cond cosine | 0.50（半長 bug）| 0.9749 |
| ss_final::shape cosine | 0.58 | 0.82 |
| ss_final::translation cosine | 0.67 | 0.998 |
| voxels jaccard | 0.10 | **0.901** |
| GS decoder fdc cosine（PT slat 輸入）| — | 0.99999 |
| Final fdc B-std vs PT 0.039 | 0.012 | **0.043** |

---

## 根本教訓

1. **Memory cap 不能武斷**：原本「16k 是 SLAT memory ceiling」假設過度保守，M1 Max 實測 26k voxels SLAT 完全跑得動
2. **Prune 不是良性 op**：surface prune + random subsample 看似保留結構，實際上系統性砍掉細節區
3. **Bottleneck 比想像深**：花 4 小時誤以為 mask DINO / fp16 / scaling_clamp 是元兇，最後發現是 pipeline 第 1027 行的硬編碼 cap
4. **逐層 dump 對比是對的方法**：餵 PT 中間結果到 MLX 各模組，能精準定位 GS decoder 沒 bug、SLAT 接近、prune 是兇手

---

## 下一步建議

- [ ] 把 `MLX_SS_MAX_VOXELS=100000` 設為 fp16 模式預設
- [ ] 把 `SLAT_GS_VARIANT=gs` (gs_32) 設為高品質模式預設（CLI 加 `--quality high`）
- [ ] commit 為 `mlx-v3-eyes-fixed` tag
- [ ] 跑 chair / table 完整 sweep 確認 no-prune 不破壞它們
- [ ] push 到用戶 GitHub fork
