# Methodology and limitations

## Research question and task

Can a supervised CNN distinguish screenshots of *Doctor Strange in the Multiverse of Madness* (Marvel) from screenshots of *Aquaman* (DC)? The experiment is a **binary classification of two films**. It must not be interpreted as evidence of robust, franchise-wide Marvel/DC recognition.

## Data preparation

The Autumn 2024 project used movie screenshots collected from an online screenshot archive and organized by class. Some source images were in WEBP format and were converted to JPG with Pillow. Images were separated into training, validation and test sets by random assignment **within each class**.

| Dataset | Total images | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| First experiment | 440 (220/class) | 308 | 66 | 66 |
| Expanded experiment | 580 (290/class) | 406 | 86 | 88 |

The published figures above are from the submitted project report. Exact original image lists, split random seeds and the full archive were not included in the materials currently in the default repository branch.

## CNN architecture and training

The **final CNN scripts supplied by the project author** use 150 × 150 RGB input; three convolutional layers (32, 64 and 128 filters, each with 3×3 kernels), each followed by max-pooling; flattening; a 512-unit dense layer; and a sigmoid output neuron. They use 1/255 pixel rescaling, Adam, binary cross-entropy, and accuracy. The tested training durations in the final script set are 10, 15 and 20 epochs. **The final scripts do not include dropout or image augmentation.** Earlier experimental code and the course notes explored other configurations.

**Reproduction note:** The cleaned script in `src/train_cnn.py` matches the supplied final CNN layer structure but uses a modern dataset loader, deterministic shuffling and full-batch evaluation. The **historical scripts** in [legacy/](../legacy/README.md) preserve the original preprocessing, training and evaluation logic; their machine-specific paths were replaced with placeholders. It is not claimed to be the exact code or environment that produced the original 2024 metrics.

## Comparative methods

Five additional algorithms were evaluated: Random Forest, Decision Tree, K-Nearest Neighbors, Logistic Regression and Gradient Boosting. The course report describes repeating experiments and comparing results. The summary tables in `RESULTS.md` are transcribed from the presentation; full raw logs and fixed random seeds are not available here.

## Critical limitations

1. **Two films, not two franchises:** Class labels and film identity are perfectly aligned; the model might exploit film-specific properties.
2. **Potential near-duplicate leakage:** Screenshots from a single movie may share scenes, actors and visually near-identical consecutive frames, inflating performance under random image-level splits.
3. **Small datasets:** The 66- and 88-image test sets mean a few misclassifications can noticeably change accuracy.
4. **Only accuracy reported:** Confusion matrices, precision/recall, independent multi-film evaluation and uncertainty intervals would strengthen an assessment.
5. **Reproducibility:** The historical screenshot archive is maintained privately and not distributed here; code snippets, settings and files have been modernized for presentation but the original complete experiment has not been rerun.

## Suggested future improvements

Evaluate on **unseen films** from both franchises; use scene/group-based splits to reduce leakage; save exact split manifests and random seeds; track experiment configurations; and report a confusion matrix, precision, recall and F1 alongside test accuracy.
