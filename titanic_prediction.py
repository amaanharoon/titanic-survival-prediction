"""
Titanic Survival Prediction
============================
Models: Logistic Regression & Decision Tree
Target : Survived (0 = No, 1 = Yes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────
df = pd.read_csv("data/titanic4__1_.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 2. Preprocessing ──────────────────────────────────────────────────────
def preprocess(df):
    data = df.copy()

    # Fill missing Age with median
    data["Age"] = data["Age"].fillna(data["Age"].median())

    # Fill missing Embarked with mode
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    # Encode Sex
    le = LabelEncoder()
    data["Sex_enc"] = le.fit_transform(data["Sex"])        # male=1, female=0

    # Encode Embarked
    data["Embarked_enc"] = le.fit_transform(data["Embarked"])

    # Age groups as feature
    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[0, 18, 60, 120],
        labels=[0, 1, 2]          # 0=child, 1=adult, 2=elderly
    ).astype(int)

    # Family size
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    return data

df = preprocess(df)

FEATURES = ["Pclass", "Sex_enc", "Age", "SibSp", "Parch",
            "Fare", "Embarked_enc", "AgeGroup", "FamilySize"]
TARGET   = "Survived"

X = df[FEATURES]
y = df[TARGET]

# ── 3. Train / test split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 4. Logistic Regression ────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr  = lr.predict(X_test_sc)
y_prob_lr  = lr.predict_proba(X_test_sc)[:, 1]

lr_acc     = accuracy_score(y_test, y_pred_lr)
lr_auc     = roc_auc_score(y_test, y_prob_lr)
lr_cv      = cross_val_score(lr, scaler.transform(X), y, cv=5, scoring="accuracy").mean()

print("\n=== Logistic Regression ===")
print(f"Accuracy : {lr_acc:.4f}")
print(f"ROC-AUC  : {lr_auc:.4f}")
print(f"CV Acc   : {lr_cv:.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["Not Survived", "Survived"]))

# ── 5. Decision Tree ──────────────────────────────────────────────────────
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt  = dt.predict(X_test)
y_prob_dt  = dt.predict_proba(X_test)[:, 1]

dt_acc     = accuracy_score(y_test, y_pred_dt)
dt_auc     = roc_auc_score(y_test, y_prob_dt)
dt_cv      = cross_val_score(dt, X, y, cv=5, scoring="accuracy").mean()

print("\n=== Decision Tree ===")
print(f"Accuracy : {dt_acc:.4f}")
print(f"ROC-AUC  : {dt_auc:.4f}")
print(f"CV Acc   : {dt_cv:.4f}")
print(classification_report(y_test, y_pred_dt, target_names=["Not Survived", "Survived"]))

# ── 6. Visualisations ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Titanic Survival Prediction — Model Evaluation", fontsize=16, fontweight="bold")

BLUE, ORANGE = "#3A7DC9", "#E07B3A"

# 6a. Confusion Matrix — LR
cm_lr = confusion_matrix(y_test, y_pred_lr)
ConfusionMatrixDisplay(cm_lr, display_labels=["Not Survived", "Survived"]).plot(
    ax=axes[0, 0], colorbar=False, cmap="Blues"
)
axes[0, 0].set_title("Confusion Matrix — Logistic Regression")

# 6b. Confusion Matrix — DT
cm_dt = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(cm_dt, display_labels=["Not Survived", "Survived"]).plot(
    ax=axes[0, 1], colorbar=False, cmap="Oranges"
)
axes[0, 1].set_title("Confusion Matrix — Decision Tree")

# 6c. ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
ax = axes[0, 2]
ax.plot(fpr_lr, tpr_lr, color=BLUE,   lw=2, label=f"Logistic Reg. (AUC={lr_auc:.2f})")
ax.plot(fpr_dt, tpr_dt, color=ORANGE, lw=2, label=f"Decision Tree (AUC={dt_auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(); ax.spines[["top", "right"]].set_visible(False)

# 6d. Feature Importance — LR (coefficients)
coef = pd.Series(np.abs(lr.coef_[0]), index=FEATURES).sort_values()
ax = axes[1, 0]
coef.plot(kind="barh", ax=ax, color=BLUE, edgecolor="white")
ax.set_title("Feature Importance — Logistic Regression\n(|coefficient|)")
ax.set_xlabel("|Coefficient|"); ax.spines[["top", "right"]].set_visible(False)

# 6e. Feature Importance — DT
imp = pd.Series(dt.feature_importances_, index=FEATURES).sort_values()
ax = axes[1, 1]
imp.plot(kind="barh", ax=ax, color=ORANGE, edgecolor="white")
ax.set_title("Feature Importance — Decision Tree")
ax.set_xlabel("Importance"); ax.spines[["top", "right"]].set_visible(False)

# 6f. Model comparison bar
ax = axes[1, 2]
metrics = ["Accuracy", "ROC-AUC", "CV Accuracy"]
lr_vals = [lr_acc, lr_auc, lr_cv]
dt_vals = [dt_acc, dt_auc, dt_cv]
x = np.arange(len(metrics)); w = 0.35
b1 = ax.bar(x - w/2, lr_vals, w, label="Logistic Reg.", color=BLUE,   edgecolor="white")
b2 = ax.bar(x + w/2, dt_vals, w, label="Decision Tree", color=ORANGE, edgecolor="white")
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
ax.set_title("Model Comparison"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/model_evaluation.png", dpi=150, bbox_inches="tight")
print("\nSaved: outputs/model_evaluation.png")

# ── 7. Decision Tree diagram ──────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=FEATURES, class_names=["Not Survived", "Survived"],
          filled=True, rounded=True, max_depth=3, ax=ax2, fontsize=9)
ax2.set_title("Decision Tree Structure (depth ≤ 3)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/decision_tree_plot.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/decision_tree_plot.png")
plt.close("all")
print("\nDone.")
