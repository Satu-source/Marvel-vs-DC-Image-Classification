# Marvel vs. DC — Image Classification with CNNs

**Deep Learning course project | Dongguk University, Seoul | Autumn 2024**  
**Satu Honka**

> A supervised learning experiment to distinguish screenshots from *Doctor Strange in the Multiverse of Madness* (Marvel) and *Aquaman* (DC), comparing a convolutional neural network (CNN) with conventional image classifiers.

**Scope:** This is a **two-film classification experiment** — not a general-purpose model for identifying the Marvel or DC franchise in arbitrary images.

## Project at a glance

| | |
| --- | --- |
| **Task** | Binary image classification: DC vs. Marvel |
| **Primary model** | Convolutional Neural Network (CNN) |
| **Tools** | Python, TensorFlow/Keras, NumPy, Pillow, scikit-learn |
| **Dataset sizes studied** | 440 and 580 movie screenshots |
| **Training design** | Approximately 70% training, 15% validation, 15% test; equal image counts per film |
| **Benchmarks** | Decision Tree, Random Forest, K-Nearest Neighbors, Logistic Regression, Gradient Boosting |
| **Highest reported CNN test accuracy** | **93.75%** on the 580-image dataset at 20 epochs |

The metrics are **historical results from the 2024 course presentation and report**, not freshly reproduced results from the reference code in this repository.

## Project workflow

1. Collect screenshots from the two films, with one class per film.
2. Convert incompatible image formats and organize the datasets.
3. Randomly divide each class into training, validation and testing sets.
4. Build and train CNN variations, comparing different epoch counts and dataset sizes.
5. Evaluate test performance and compare with five classical ML approaches.
6. Document methodological choices, limitations and lessons learned.

## Selected CNN results

| Images | Epochs | Reported test accuracy |
| ---: | ---: | ---: |
| 440 | 10 | 89.95% |
| 440 | **15** | **91.02%** |
| 440 | 20 | 89.84% |
| 580 | 10 | 90.11% |
| 580 | 15 | 92.69% |
| 580 | **20** | **93.75%** |

The best classical comparison result recorded was **90.91% test accuracy with K-Nearest Neighbors**. See [full experiment tables](docs/RESULTS.md).

## Code and documentation

| Resource | Description |
| --- | --- |
| [CNN reference implementation](src/train_cnn.py) | Standalone, cleaned-up *example based on the original model structure*; not the exact archived training run |
| [Methodology](docs/METHODOLOGY.md) | Data splitting, model choices, evaluation approach and methodological limitations |
| [Results](docs/RESULTS.md) | CNN experiments and conventional ML comparisons |
| [Original comparison methods](legacy/other_techniques/README.md) | Five archived scikit-learn baselines, their hyperparameters and image-processing workflow |
| [Unused historical CNN experiments](legacy/unused_experiments/README.md) | Early prototypes retained as development history; **not** sources of reported results |
| [Dataset notes](docs/DATASET.md) | Image sources, licensing considerations and incomplete GitHub archive |
| [Historical final CNN & preprocessing scripts](legacy/README.md) | The original 10/15/20-epoch final versions and data-preparation scripts; paths have been anonymized | 
| [Original format-conversion prototype](Image%20format%20change) | Previously committed conversion draft; machine-specific paths have been anonymized |
| Movie screenshot dataset | **Not distributed** in this public repository. Image files are retained separately in the author's private project archive. |

## Run the reference example

You will need your **own legally usable image dataset** organized like this:

```text
data/
├── train/
│   ├── DC/
│   └── Marvel/
├── valid/
│   ├── DC/
│   └── Marvel/
└── test/
    ├── DC/
    └── Marvel/
```

```bash
python -m pip install -r requirements.txt
python src/train_cnn.py --data-dir data --epochs 15
```

The script follows the final project's 150 × 150 input and three-convolution architecture, but uses a modern data-loading API and processes all batches. It has not been tested against the original 2024 dataset. It outputs metrics and an example saved model in `outputs/`. It has **not been re-run with the original 2024 dataset**, and results may differ.

## Lessons and limitations

- Image conversion, consistent class labels and clean dataset directories were necessary to make training work.
- The project examined both dataset size and training duration; more epochs did not always improve test accuracy.
- **Film identity is confounded with franchise identity:** because each class comes from one film, the classifier might learn movie-specific colors, characters and cinematography rather than general Marvel/DC characteristics.
- Randomly splitting screenshots from the same films can create overly similar frames across train and test sets, making generalization look stronger than it would be on unseen films.
- The privately retained screenshot archive is an archival **subset**, not evidence that the original experiment used only those images.

## Public release and dataset policy

This public repository contains code and text documentation only. **No copyrighted movie screenshots have been copied into this public repository or its commit history.** The image data from the educational experiment is maintained separately in a private archive.

## Credits and image rights

Project developed by **Satu Honka** during the Deep Learning course at **Dongguk University (Autumn 2024)**, while studying at Häme University of Applied Sciences (HAMK). The images used in the study are screenshots from third-party copyrighted films; rights remain with their respective owners. The data is **not licensed for unrestricted redistribution**. Please supply your own authorized images when reproducing the example.

*This repository documents an educational proof of concept, not a production-ready classifier.*
