# PolymerCode AI — Classification Crops

This folder contains single-symbol crops prepared from the Roboflow Plastic Recycle Sign dataset.

## Source and attribution
- Source: Plastic Recycle Sign by Anthony, Roboflow Universe: https://universe.roboflow.com/anthony-q0mmp/plastic_recycle_sign
- License shown in the Roboflow export: CC BY 4.0. Attribute the original dataset author/source in project reports and redistribute under the applicable license terms.

## What this data contains
- Images: 536 crop files, derived from annotated YOLOv8 boxes.
- Crop resolution: 224 × 224 RGB.
- Crop padding: 12% around the bounding box; aspect ratio preserved with reflected-edge padding; no geometric stretching of each crop.
- No augmentation has been applied.
- Class folders: HDPE, LDPE, OTHER, PET, PP, PS, PVC.
- `metadata.csv` links each crop to the original image, original split, box, class and new split.

## Class to resin-code mapping
- HDPE: resin code 2
- LDPE: resin code 4
- OTHER: resin code 7
- PET: resin code 1
- PP: resin code 5
- PS: resin code 6
- PVC: resin code 3

## Splits
A deterministic stratified 20-fold assignment (seed 42) uses folds 0–2 as test, 3–5 as validation, and 6–19 as training, approximately 70/15/15. These splits intentionally supersede the Roboflow train/validation/test assignments so rare classes are represented in each split. They are at the original-image level; no augmented variants are present.

## Important limitations
The classes are imbalanced, particularly PVC and PS. Use class weighting or balanced sampling only on training data and report macro-F1 plus per-class metrics. The current test set is still small for rare classes; avoid overclaiming results. Some embossed symbols are low contrast or visually ambiguous, so inspect samples before modeling.

## Directory layout
`train/<CLASS>/*.jpg`, `val/<CLASS>/*.jpg`, and `test/<CLASS>/*.jpg` are ready for a Keras/PyTorch image-classification loader.
