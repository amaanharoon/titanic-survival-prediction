"""
Titanic Survival — Exploratory Data Analysis (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("data/titanic4__1_.csv")

print("=== Dataset Info ===")
print(f"Shape : {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nBasic stats:")
print(df.describe())

# ── Age groups ────────────────────────────────────────────────────────────
bins   = [0, 18, 60, 120]
labels = ["Children (<18)", "Adults (18–59)", "Elderly (60+)"]
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

# ── Survival rates ────────────────────────────────────────────────────────
overall   = df["Survived"].mean() * 100
male_r    = df[df["Sex"] == "male"]["Survived"].mean() * 100
female_r  = df[df["Sex"] == "female"]["Survived"].mean() * 100
age_surv  = df.groupby("AgeGroup", observed=True)["Survived"].mean() * 100
gender_age = df.groupby(["Sex", "AgeGroup"], observed=True)["Survived"].mean() * 100

print(f"\nOverall survival : {overall:.1f}%")
print(f"Male             : {male_r:.1f}%")
print(f"Female           : {female_r:.1f}%")
print("\nBy Age Group:")
print(age_surv.to_string())

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic — Exploratory Data Analysis", fontsize=16, fontweight="bold")

PINK, BLUE = "#E05C97", "#3A7DC9"
GREEN, AMBER, PURPLE = "#4CAF50", "#FF9800", "#9C27B0"

# 1. Overall survival
ax = axes[0, 0]
counts = df["Survived"].value_counts()
ax.pie(counts, labels=["Not Survived", "Survived"], colors=["#ddd", BLUE],
       autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Overall Survival")

# 2. Gender survival
ax = axes[0, 1]
gender = df.groupby("Sex")["Survived"].mean() * 100
bars = ax.bar(["Male", "Female"], [gender["male"], gender["female"]],
              color=[BLUE, PINK], edgecolor="white", width=0.5)
ax.axhline(overall, color="gray", linestyle="--", linewidth=1.2, label=f"Overall {overall:.1f}%")
for bar, val in zip(bars, [gender["male"], gender["female"]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 100); ax.set_ylabel("Survival Rate (%)")
ax.set_title("Survival Rate by Gender"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# 3. Age group survival
ax = axes[0, 2]
bars = ax.bar(age_surv.index.astype(str), age_surv.values,
              color=[GREEN, AMBER, PURPLE], edgecolor="white", width=0.5)
for bar, val in zip(bars, age_surv.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 100); ax.set_ylabel("Survival Rate (%)")
ax.set_title("Survival Rate by Age Group")
ax.tick_params(axis="x", rotation=10)
ax.spines[["top", "right"]].set_visible(False)

# 4. Gender × Age Group
ax = axes[1, 0]
x = np.arange(len(labels)); w = 0.35
male_vals   = [gender_age.get(("male",   g), 0) for g in labels]
female_vals = [gender_age.get(("female", g), 0) for g in labels]
b1 = ax.bar(x - w/2, male_vals,   w, label="Male",   color=BLUE,  edgecolor="white")
b2 = ax.bar(x + w/2, female_vals, w, label="Female", color=PINK,  edgecolor="white")
for bars_, vals in [(b1, male_vals), (b2, female_vals)]:
    for bar, val in zip(bars_, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n") for l in labels])
ax.set_ylim(0, 115); ax.set_ylabel("Survival Rate (%)")
ax.set_title("Gender × Age Group"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# 5. Age distribution
ax = axes[1, 1]
survived     = df[df["Survived"] == 1]["Age"].dropna()
not_survived = df[df["Survived"] == 0]["Age"].dropna()
ax.hist(not_survived, bins=20, alpha=0.6, color="#ddd",  label="Not Survived", edgecolor="white")
ax.hist(survived,     bins=20, alpha=0.7, color=BLUE,    label="Survived",     edgecolor="white")
ax.set_xlabel("Age"); ax.set_ylabel("Count")
ax.set_title("Age Distribution by Survival")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)

# 6. Pclass survival
ax = axes[1, 2]
pclass = df.groupby("Pclass")["Survived"].mean() * 100
bars = ax.bar(["1st Class", "2nd Class", "3rd Class"], pclass.values,
              color=[BLUE, AMBER, PINK], edgecolor="white", width=0.5)
for bar, val in zip(bars, pclass.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 100); ax.set_ylabel("Survival Rate (%)")
ax.set_title("Survival Rate by Passenger Class")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/eda_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved: outputs/eda_analysis.png")
plt.close()
print("Done.")
