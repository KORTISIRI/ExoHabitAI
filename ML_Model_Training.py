import os
import warnings
from typing import List

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from exohabit_pipeline import (
    PLOTS_DIR,
    PREPROCESSED_DATA_PATH,
    PROCESSED_DATA_PATH,
    RANKED_DATA_PATH,
    TARGET_COLUMN,
    artifact_paths,
    blend_probabilities,
    ensure_directories,
    get_feature_columns,
    load_raw_dataset,
    preprocess_raw_data,
    save_feature_metadata,
)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    categorical_features = [column for column in feature_columns if column == "st_spectype"]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_pipeline(classifier, feature_columns: List[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_columns)),
            ("clf", classifier),
        ]
    )


def build_models(scale_pos_weight: float, feature_columns: List[str]) -> dict:
    models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                C=0.75,
                class_weight="balanced",
                max_iter=2500,
                random_state=RANDOM_STATE,
            ),
            feature_columns,
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            feature_columns,
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            feature_columns,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = make_pipeline(
            XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            feature_columns,
        )

    return models


def evaluate_models(models: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    predictions = {}
    probabilities = {}

    print("=" * 100)
    print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'CV F1':>8} {'Status':>10}")
    print("=" * 100)

    for name, model in models.items():
        model.fit(X_train, y_train)
        raw_probabilities = model.predict_proba(X_test)[:, 1]
        blended_probabilities = blend_probabilities(raw_probabilities, X_test)
        predicted_labels = (blended_probabilities >= 0.5).astype(int)

        metrics = {
            "acc": accuracy_score(y_test, predicted_labels),
            "prec": precision_score(y_test, predicted_labels, zero_division=0),
            "rec": recall_score(y_test, predicted_labels, zero_division=0),
            "f1": f1_score(y_test, predicted_labels, zero_division=0),
            "auc": roc_auc_score(y_test, blended_probabilities),
        }
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()
        metrics["status"] = "stable" if abs(metrics["f1"] - metrics["cv_mean"]) <= 0.05 else "overfit"

        print(
            f"{name:<22} {metrics['acc']:>7.4f} {metrics['prec']:>7.4f} {metrics['rec']:>7.4f} "
            f"{metrics['f1']:>7.4f} {metrics['auc']:>7.4f} {metrics['cv_mean']:>8.4f} {metrics['status']:>10}"
        )

        results[name] = metrics
        predictions[name] = predicted_labels
        probabilities[name] = blended_probabilities

    return results, predictions, probabilities


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, feature_columns: List[str]) -> RandomizedSearchCV:
    estimator = make_pipeline(
        RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        feature_columns,
    )
    params = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [8, 12, 16, None],
        "clf__min_samples_leaf": [2, 4, 8],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=20,
        scoring="roc_auc",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float, feature_columns: List[str]):
    estimator = make_pipeline(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        feature_columns,
    )
    params = {
        "clf__n_estimators": [200, 300, 400],
        "clf__max_depth": [4, 5, 6],
        "clf__learning_rate": [0.03, 0.05, 0.07],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__gamma": [0.0, 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=15,
        scoring="roc_auc",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def composite_score(metrics: dict) -> float:
    stability_penalty = 0.08 if abs(metrics["cv_mean"] - metrics["f1"]) > 0.08 else 0.0
    return (
        0.45 * metrics["f1"]
        + 0.30 * metrics["rec"]
        + 0.15 * metrics["auc"]
        + 0.10 * metrics["prec"]
        - stability_penalty
    )


def feature_importance_frame(model: Pipeline) -> pd.DataFrame | None:
    classifier = model.named_steps["clf"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in preprocessor.get_feature_names_out()
    ]

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        return None

    return pd.DataFrame({"Feature": feature_names, "Importance": values}).sort_values(
        "Importance", ascending=False
    )


def main() -> None:
    ensure_directories()
    paths = artifact_paths()

    print("=" * 60)
    print("ExoHabitAI - ML Model Training")
    print("=" * 60)

    raw_df = load_raw_dataset()
    preprocessed_df, model_df, preprocessing_profile = preprocess_raw_data(raw_df)
    feature_columns = get_feature_columns(model_df)
    save_feature_metadata(feature_columns, preprocessing_profile)

    preprocessed_df.to_csv(PREPROCESSED_DATA_PATH, index=False)
    model_df.to_csv(PROCESSED_DATA_PATH, index=False)

    X = model_df[feature_columns]
    y = model_df[TARGET_COLUMN]
    print(f"Dataset shape: {model_df.shape}")
    print(f"Habitable planets: {int(y.sum())}")
    print(f"Non-habitable planets: {int((y == 0).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = build_models(scale_pos_weight, feature_columns)
    print(f"Training models: {list(models.keys())}")

    results, predictions, probabilities = evaluate_models(models, X_train, X_test, y_train, y_test)

    best_rf_search = tune_random_forest(X_train, y_train, feature_columns)
    best_rf = best_rf_search.best_estimator_
    rf_raw_prob = best_rf.predict_proba(X_test)[:, 1]
    rf_prob = blend_probabilities(rf_raw_prob, X_test)
    rf_pred = (rf_prob >= 0.5).astype(int)
    rf_cv_scores = cross_val_score(best_rf, X_train, y_train, cv=CV_FOLDS, scoring="f1", n_jobs=-1)
    results["Random Forest Tuned"] = {
        "acc": accuracy_score(y_test, rf_pred),
        "prec": precision_score(y_test, rf_pred, zero_division=0),
        "rec": recall_score(y_test, rf_pred, zero_division=0),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
        "auc": roc_auc_score(y_test, rf_prob),
        "cv_mean": rf_cv_scores.mean(),
        "cv_std": rf_cv_scores.std(),
        "status": "tuned",
    }
    predictions["Random Forest Tuned"] = rf_pred
    probabilities["Random Forest Tuned"] = rf_prob
    print(f"Tuned Random Forest best params: {best_rf_search.best_params_}")

    tuned_xgb = None
    if XGBOOST_AVAILABLE:
        best_xgb_search = tune_xgboost(X_train, y_train, scale_pos_weight, feature_columns)
        tuned_xgb = best_xgb_search.best_estimator_
        xgb_raw_prob = tuned_xgb.predict_proba(X_test)[:, 1]
        xgb_prob = blend_probabilities(xgb_raw_prob, X_test)
        xgb_pred = (xgb_prob >= 0.5).astype(int)
        xgb_cv_scores = cross_val_score(tuned_xgb, X_train, y_train, cv=CV_FOLDS, scoring="f1", n_jobs=-1)
        results["XGBoost Tuned"] = {
            "acc": accuracy_score(y_test, xgb_pred),
            "prec": precision_score(y_test, xgb_pred, zero_division=0),
            "rec": recall_score(y_test, xgb_pred, zero_division=0),
            "f1": f1_score(y_test, xgb_pred, zero_division=0),
            "auc": roc_auc_score(y_test, xgb_prob),
            "cv_mean": xgb_cv_scores.mean(),
            "cv_std": xgb_cv_scores.std(),
            "status": "tuned",
        }
        predictions["XGBoost Tuned"] = xgb_pred
        probabilities["XGBoost Tuned"] = xgb_prob
        print(f"Tuned XGBoost best params: {best_xgb_search.best_params_}")

    best_name = max(results, key=lambda name: composite_score(results[name]))
    tuned_models = {
        "Random Forest Tuned": best_rf,
        "XGBoost Tuned": tuned_xgb,
    }
    best_model = tuned_models.get(best_name) or models[best_name]
    best_metrics = results[best_name]

    print("=" * 60)
    print("Model Selection")
    print("=" * 60)
    print(f"Winner: {best_name}")
    print(
        f"F1={best_metrics['f1']:.4f}, Recall={best_metrics['rec']:.4f}, "
        f"AUC={best_metrics['auc']:.4f}, CV F1={best_metrics['cv_mean']:.4f}"
    )

    joblib.dump(models["Logistic Regression"], paths["logistic_regression"])
    joblib.dump(models["Decision Tree"], paths["decision_tree"])
    joblib.dump(models["Random Forest"], paths["random_forest"])
    joblib.dump(best_rf, paths["random_forest_tuned"])
    if XGBOOST_AVAILABLE and "XGBoost" in models:
        joblib.dump(models["XGBoost"], paths["xgboost"])
    if tuned_xgb is not None:
        joblib.dump(tuned_xgb, paths["xgboost_tuned"])
    joblib.dump(best_model, paths["final_model"])
    joblib.dump(results, paths["training_summary"])

    print("Saved model artifacts")

    report = classification_report(
        y_test,
        predictions[best_name],
        target_names=["Non-Habitable", "Habitable"],
        zero_division=0,
    )
    print("=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    cm = confusion_matrix(y_test, predictions[best_name])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Habitable", "Habitable"],
        yticklabels=["Non-Habitable", "Habitable"],
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 7))
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        width = 2 if name == best_name else 1
        plt.plot(fpr, tpr, linewidth=width, label=f"{name} (AUC={results[name]['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    full_probabilities = blend_probabilities(best_model.predict_proba(X)[:, 1], X)
    ranked_df = model_df.copy()
    ranked_df["habitability_probability"] = full_probabilities
    ranked_df["predicted_label"] = (full_probabilities >= 0.5).astype(int)
    ranked_df = ranked_df.sort_values("habitability_probability", ascending=False).reset_index(drop=True)
    ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))
    ranked_df.to_csv(RANKED_DATA_PATH, index=False)

    importance_df = feature_importance_frame(best_model)
    if importance_df is not None:
        importance_df.to_csv("data/processed/feature_importances.csv", index=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", hue="Feature", palette="Blues_r")
        plt.legend([], [], frameon=False)
        plt.title(f"Top 10 Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
        plt.close()
        print("Top 5 features:")
        print(importance_df.head(5).to_string(index=False))

    comparison_df = (
        pd.DataFrame(results)
        .transpose()
        .sort_values(["f1", "rec", "auc"], ascending=False)
        .reset_index()
        .rename(columns={"index": "model"})
    )
    comparison_df.to_csv("data/processed/model_comparison.csv", index=False)

    print("=" * 60)
    print("Training pipeline completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
