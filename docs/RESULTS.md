# Recorded experimental results — Autumn 2024

These numbers are transcribed from Satu Honka's **final Deep Learning course presentation** and were reported as part of the original project. They are **not new model evaluations performed for this repository**.

## CNN comparisons

| Total images | Epochs | Validation accuracy | Test accuracy | Reported runtime (s) |
| ---: | ---: | ---: | ---: | ---: |
| 440 | 10 | 100.00% | 89.95% | 24.32 |
| 440 | 15 | 92.19% | **91.02%** | 36.64 |
| 440 | 20 | 92.97% | 89.84% | 45.06 |
| 580 | 10 | 93.18% | 90.11% | 28.62 |
| 580 | 15 | 91.41% | 92.69% | 44.07 |
| 580 | 20 | 85.73% | **93.75%** | 56.62 |

The strongest reported **CNN test result** was **93.75%** with 580 images and 20 epochs. The best test result with 440 images was **91.02%**, achieved with 15 epochs.

## Conventional machine-learning comparisons

| Model | Test accuracy: 440 images | Test accuracy: 580 images |
| --- | ---: | ---: |
| Random Forest | 86.36% | 86.65% |
| Decision Tree | 73.87% | 75.85% |
| K-Nearest Neighbors (KNN) | **90.91%** | **90.91%** |
| Logistic Regression | 86.36% | 88.64% |
| Gradient Boosting | 87.88% | 90.63% |

The comparison did not establish that CNNs are universally better; it only describes the tested settings and these two-film datasets.

## Interpretation

- Increasing the dataset from 440 to 580 images was associated with a higher best reported CNN test accuracy, but **performance was not monotonic** across epoch counts.
- KNN was a comparatively strong non-CNN baseline.
- Large differences between validation and test accuracy in some settings merit investigation.
- Runtime figures are specific to the project's environment and are not comparable to modern hardware benchmarks.

The underlying image-level predictions and saved model checkpoints were not provided with these summary tables. See [methodological limitations](METHODOLOGY.md).

## Implementation of the five non-CNN baselines

The five [archived original scripts](../legacy/other_techniques/README.md) implement **Decision Tree, Random Forest (100 trees), KNN (5 neighbors), Logistic Regression (max_iter=200)** and **Gradient Boosting (100 estimators)**. They convert each 150 × 150 RGB image into a flattened **67,500-value raw-pixel vector** and evaluate the resulting scikit-learn classifiers on validation and test subsets. They do not extract CNN image embeddings. The reported accuracies above are taken from the project documentation and have **not** been reproduced by running the uploaded scripts.
