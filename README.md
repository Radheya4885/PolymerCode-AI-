# PolymerCode AI

Artificial Neural Networks (ANN) course project for plastic resin-code
classification from images.

## Package contents
- notebooks/: complete Colab notebook
- data/processed/classification_dataset/: image crops, splits and metadata
- models/: three PyTorch checkpoints and three PKL state-dict bundles
- models/model_manifest.json: class mapping and model configuration
- src/: reusable model architecture definitions
- results/: available training histories, reports and plots
- frontend/ and backend/: placeholders for Antigravity implementation

## Resin codes
PET=1, HDPE=2, PVC=3, LDPE=4, PP=5, PS=6, OTHER=7.

## Checkpoints
- custom_cnn_best.pt / custom_cnn.pkl
- vgg16_best.pt / vgg16.pkl
- vgg16_balanced_best.pt / vgg16_balanced.pkl

The PT checkpoints are the primary PyTorch artifacts. The PKL files hold
state dictionaries and configuration, not complete serialized model objects.
Rebuild the matching architecture from src/model_architectures.py before
loading weights. Only load pickle files from trusted sources.

## Reported initial results
Custom CNN test: accuracy 35.80%, macro-F1 0.2532.
Original VGG-16 test: accuracy 61.73%, macro-F1 0.5436.
Balanced VGG-16 validation macro-F1: 0.4413.
The balanced VGG-16 has not yet been evaluated on the test set.

The dataset contains 536 crops and seven classes. PS and PVC have few
examples, so their evaluation scores are uncertain. Model scores are not
calibrated probabilities and predictions may be incorrect.
