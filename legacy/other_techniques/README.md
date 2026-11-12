# Other techniques — classical machine learning baselines

These are the **five original comparison scripts** from Satu Honka's Autumn 2024 Deep Learning project at Dongguk University. They are separate from the final CNN training scripts and were used to evaluate alternative approaches to the same binary film-screenshot classification task.

| Method | Original script | Scikit-learn model |
| --- | --- | --- |
| Decision Tree | [Classification_version5_desicion_tree.py](Classification_version5_desicion_tree.py) | `DecisionTreeClassifier()` |
| Random Forest | [Classification_version5_random_forest.py](Classification_version5_random_forest.py) | `RandomForestClassifier(n_estimators=100)` |
| K-Nearest Neighbors | [Classification_version5_k_nearestNeighborn.py](Classification_version5_k_nearestNeighborn.py) | `KNeighborsClassifier(n_neighbors=5)` |
| Logistic Regression | [Classification_version5_logistic_regressio.py](Classification_version5_logistic_regressio.py) | `LogisticRegression(max_iter=200)` |
| Gradient Boosting | [Classification_version5_GBM.py](Classification_version5_GBM.py) | `GradientBoostingClassifier(n_estimators=100)` |

The filenames preserve the original project spelling, including the historical typos.

## Shared processing approach

1. Images are loaded using TensorFlow/Keras `ImageDataGenerator(rescale=1./255)`.
2. `flow_from_directory` reads `Combined/train`, `Combined/valid` and `Combined/test` at **150 × 150 pixels**, with `shuffle=False` and batch size 32.
3. Each RGB image is reshaped into a **67,500-dimensional flattened raw-pixel feature vector** (150 × 150 × 3). These are *not* CNN-generated embeddings or feature maps.
4. Each scikit-learn classifier is fitted to the training features and labels.
5. Validation accuracy, test accuracy and execution time are printed.

## Reproducibility and methodological notes

- The original files have been **archived, not rewritten or re-executed**. Only absolute Windows data paths were replaced by `REPLACE_WITH_PROJECT_ROOT` and a short header was added.
- The original screenshot dataset is not bundled with these scripts. Local paths must be adjusted and suitable images provided before running them.
- Most estimators use default hyperparameters; the scripts generally do **not** fix a random seed or perform systematic hyperparameter tuning.
- Using high-dimensional raw pixel vectors rather than engineered or pretrained image features affects the interpretation of this comparison.
- Each class comes from one movie, and images were randomly split within movies; similar frames may occur across subsets, so results should not be generalized to entirely unseen films.
- Reported results are shown in [the original project's results summary](../../docs/RESULTS.md). They are historical figures, not rerun benchmarks.

**Original source provenance:** Files provided by the author as part of the course's *Other techniques* folder.
