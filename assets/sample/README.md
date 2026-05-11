# Sample input

`cat.jpg` — orange tabby photo by [Sam Lion](https://www.pexels.com/@samson-katt) on
[Pexels](https://www.pexels.com/photo/orange-tabby-cat-with-red-bandana-1741205/),
distributed under the [Pexels License](https://www.pexels.com/license/) (free
for commercial use, redistribution allowed, no attribution required).

`cat_mask.png` — binary foreground mask generated with
[rembg](https://github.com/danielgatis/rembg) (U2Net) and shipped here so you
can run the Quickstart without setting up segmentation tooling.

## Use

```bash
python meadow_wb/infer.py \
    --image assets/sample/cat.jpg \
    --mask  assets/sample/cat_mask.png \
    --use-moge --use-shortcut --dtype mixed --prune-outliers \
    --out outputs/cat.ply
```

Expected end-to-end wall on M1 Max with v0.0.2 defaults: **~40 s** (slightly
slower than the chair/table/plush ~31 s because the cat's voxel count after
SS is higher).
