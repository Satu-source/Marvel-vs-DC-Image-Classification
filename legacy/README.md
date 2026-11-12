# Historical source code — Autumn 2024

These files document the actual source scripts provided by the project author for the **final CNN experiments** and their **image preparation** steps. They were used as the source basis for the repository's modernized [CNN reference script](../src/train_cnn.py).

**Only the machine-specific absolute path strings were anonymized** to `REPLACE_WITH_PROJECT_ROOT`, and an explanatory header was added to each archival file. Otherwise, the original code has been retained; it has **not been refactored or rerun** here.

## Final CNN experiment scripts

| File | Training epochs | Model structure |
| --- | ---: | --- |
| [Classification_version4_epoch10.py](final_cnn/Classification_version4_epoch10.py) | 10 | Three convolution/pooling blocks; Dense 512; sigmoid output |
| [Classification_version4_epoch15.py](final_cnn/Classification_version4_epoch15.py) | 15 | Same architecture |
| [Classification_version4_epoch20.py](final_cnn/Classification_version4_epoch20.py) | 20 | Same architecture |

All three scripts load 150 × 150 RGB images using `ImageDataGenerator(rescale=1./255)`, compile with Adam / binary cross-entropy / accuracy, train using a batch size of 32 and evaluate on held-out test images.

**Key observation:** The final three scripts differ only in the number of epochs and accompanying comments. Their architecture and data loading code are otherwise the same.

## Image preparation

| File | Purpose |
| --- | --- |
| [image_format_change.py](data_preparation/image_format_change.py) | Convert WEBP image files to JPEG using Pillow |
| [image splitting.py](data_preparation/image%20splitting.py) | Randomly split Marvel and DC image files into train/validation/test folders, using 70/15/15 proportions |
| [Connecting training set.py](data_preparation/Connecting%20training%20set.py) | Copy the separate class folders into a combined split/class directory layout |

The historical scripts are **not intended as a turn-key reproduction** because the original local file paths were replaced and the complete copyrighted screenshot dataset has not been included. Use the cleaned [reference script](../src/train_cnn.py) with your own suitable image dataset for a portable example.

### Technical limitations of the historical scripts

- The image split was random, but no fixed random seed was specified in the supplied split script.
- Each model's `steps_per_epoch`, `validation_steps`, and test evaluation `steps` use floor division by 32. This can omit the final partial batch in the reported evaluation.
- The `flow_from_directory` calls use default shuffling behavior; for rigorous evaluation, use deterministic test-set ordering and preserve label/prediction alignment.
- A class consists of screenshots from only one film, so film-specific visual patterns may confound a franchise-level interpretation.

## Experimental and baseline files

The original **[“codes I didn't use” archive](unused_experiments/README.md)** contains five unsuccessful or exploratory CNN versions; they are separated from the final reported experiments.

The five original **[Other techniques](other_techniques/README.md)** comparison scripts have now been archived separately: Decision Tree, Random Forest, KNN, Logistic Regression and Gradient Boosting. Their reported results are summarized in [RESULTS.md](../docs/RESULTS.md).
