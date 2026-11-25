"""Compare Logistic Regression, Decision Tree and Random Forest on German Credit."""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import CATEGORICAL, prepare

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/german_credit_data.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SEED = 42


def build_preprocessor(X):
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def main():
    X, y = prepare(DATA)
    print(f"Rows: {len(X)} | Bad-risk rate: {y.mean():.2%}")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=SEED),
    }
    models = {n: Pipeline([("prep", build_preprocessor(X)), ("clf", c)])
              for n, c in classifiers.items()}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, curves = [], {}
    for name, m in models.items():
        cv_auc = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        train_auc = roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])
        test_auc = roc_auc_score(y_te, proba)
        print(f"{name}: train AUC={train_auc:.3f} | test AUC={test_auc:.3f}")
        rows.append({
            "Model": name,
            "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred),
            "F1": f1_score(y_te, pred),
            "ROC-AUC": test_auc,
            "Train AUC": train_auc,
            "CV ROC-AUC (mean)": cv_auc.mean(),
            "CV ROC-AUC (std)": cv_auc.std(),
        })
        curves[name] = roc_curve(y_te, proba)
        print(f"{name} confusion matrix [[TN FP] [FN TP]]:\n{confusion_matrix(y_te, pred)}\n")

    results = pd.DataFrame(rows).round(3)
    print(results.to_string(index=False))
    results.to_csv(OUT / "results.csv", index=False)

    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, _) in curves.items():
        auc = results.loc[results.Model == name, "ROC-AUC"].iloc[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (test set)"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=150); plt.close()

    rf = models["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imps = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(15)
    plt.figure(figsize=(7, 6))
    imps.plot.barh(); plt.title("Random Forest - Top 15 Features")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()

    lr = models["Logistic Regression"]
    coefs = pd.Series(lr.named_steps["clf"].coef_[0],
                      index=lr.named_steps["prep"].get_feature_names_out()).sort_values()
    coefs.to_csv(OUT / "logistic_coefficients.csv", header=["coefficient"])

    joblib.dump(models["Random Forest"], OUT / "model.joblib")
    print("\nSaved results and model to outputs/")


if __name__ == "__main__":
    main()
