python -m pip install numpy scikit-learn
"""
Phase 2 - Option 1: Baseline Implementation
AJA.NIBRAZ | KIC-BSCCSAI-251P-055

Breast cancer classification using Logistic Regression.

Dataset:
https://doi.org/10.24432/C5DW2B

Research paper:
https://doi.org/10.1016/j.procs.2021.07.062

Run:
    python baseline.py
"""

from pathlib import Path
import csv
import json
import platform

import numpy as np
import sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
THRESHOLD = 0.5
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the bundled Wisconsin Diagnostic dataset.
    dataset = load_breast_cancer()
    X = dataset.data

    # Encode malignant as the positive class:
    # malignant = 1, benign = 0.
    y = (dataset.target == 0).astype(int)

    # 2. Check dataset structure and numerical values.
    if X.shape != (569, 30):
        raise ValueError("Expected 569 records and 30 features.")

    if not np.isfinite(X).all():
        raise ValueError("Missing or non-finite values found.")

    print("Records:", len(y))
    print("Features:", X.shape[1])
    print("Benign:", int(np.sum(y == 0)))
    print("Malignant:", int(np.sum(y == 1)))

    # 3. Reserve 25% of records for final testing.
    # Stratification preserves approximate class proportions.
    indices = np.arange(len(y))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    assert not set(train_indices) & set(test_indices)

    # 4. Combine scaling and classification.
    # During cross-validation, scaling is fitted only
    # on each fold's training records.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                random_state=SEED,
            ),
        ),
    ])

    # 5. Select regularization strength using only training data.
    # Smaller C means stronger regularization.
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        scoring="roc_auc",
        cv=cross_validation,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    # The selected pipeline is refitted on all training records.
    best_model = search.best_estimator_

    # 6. Predict on the held-out test records.
    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # 7. Calculate classification metrics.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "malignant_recall": float(
            recall_score(y_test, predictions)
        ),
        "specificity": float(tn / (tn + fp)),
        "f1_score": float(
            f1_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        # ROC AUC requires probabilities rather than class labels.
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # 8. Compare with published accuracy values.
    # These are reference results, not newly trained models.
    comparison = {
        "paper_logistic_regression_accuracy": 0.958,
        "paper_svm_accuracy": 0.972,
        "measured_baseline_accuracy": metrics["accuracy"],
        "difference_from_paper_lr_percentage_points": (
            metrics["accuracy"] - 0.958
        ) * 100,
        "difference_from_paper_svm_percentage_points": (
            metrics["accuracy"] - 0.972
        ) * 100,
        "limitation": (
            "Different splits and settings mean this is not "
            "an exact replication or proof of superiority."
        ),
    }

    # 9. Save results and experiment settings.
    report = {
        "model": "StandardScaler and Logistic Regression",
        "dataset": "Wisconsin Diagnostic",
        "dataset_source": "Scikit-learn bundled dataset",
        "positive_class": "malignant",
        "training_records": len(train_indices),
        "test_records": len(test_indices),
        "seed": SEED,
        "threshold": THRESHOLD,
        "selected_C": float(
            search.best_params_["classifier__C"]
        ),
        "mean_validation_roc_auc": float(search.best_score_),
        "feature_names": dataset.feature_names.tolist(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "metrics": metrics,
        "paper_comparison": comparison,
        "limitations": [
            "One dataset and one held-out test split.",
            "No external clinical validation.",
            "Phase 1 already explored the complete dataset.",
            "This is a computational holdout, not a new cohort.",
            "The script does not train SVM.",
        ],
    }

    with (OUTPUT_DIR / "results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    # 10. Save individual test predictions.
    with (OUTPUT_DIR / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset_row_index",
            "actual_label",
            "predicted_label",
            "malignant_probability",
        ])

        for index, actual, predicted, probability in zip(
            test_indices, y_test, predictions, probabilities
        ):
            writer.writerow([
                int(index),
                int(actual),
                int(predicted),
                float(probability),
            ])

    # 11. Save the exact train/test split.
    with (OUTPUT_DIR / "split_indices.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
            },
            file,
            indent=2,
        )

    # 12. Save validation scores for each candidate C.
    with (OUTPUT_DIR / "cross_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "C",
            "mean_validation_auc",
            "standard_deviation_validation_auc",
        ])

        for parameters, mean, deviation in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        ):
            writer.writerow([
                parameters["classifier__C"],
                float(mean),
                float(deviation),
            ])

    print("\nSelected C:", report["selected_C"])
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))

    print("\nPaper comparison:")
    print(json.dumps(comparison, indent=2))

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
