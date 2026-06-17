# 🚢 Titanic Survival Prediction

Predicting passenger survival on the Titanic using **Logistic Regression** and **Decision Tree** classifiers.

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── data/
│   └── titanic4__1_.csv          # Dataset (891 passengers, 12 features)
│
├── outputs/                       # Generated charts and plots
│   ├── eda_analysis.png
│   ├── model_evaluation.png
│   └── decision_tree_plot.png
│
├── eda.py                         # Exploratory Data Analysis
├── titanic_prediction.py          # Model training & evaluation
├── requirements.txt
└── README.md
```

## 📊 Dataset

| Feature | Description |
|---|---|
| `Survived` | Target — 0 = No, 1 = Yes |
| `Pclass` | Passenger class (1st / 2nd / 3rd) |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | # siblings / spouses aboard |
| `Parch` | # parents / children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation (C / Q / S) |

## 🔍 Key Findings (EDA)

| Group | Survival Rate |
|---|---|
| Overall | 38.4% |
| Female | 74.2% |
| Male | 18.9% |
| Children (<18) | 50.4% |
| Adults (18–59) | 38.9% |
| Elderly (60+) | 22.7% |

> "Women and children first" is clearly visible in the data — females survived at nearly **4× the rate** of males.

## 🤖 Models

### Logistic Regression
- Scaled features with `StandardScaler`
- `max_iter=1000`, `random_state=42`
- Evaluated with Accuracy, ROC-AUC, and 5-fold CV

### Decision Tree
- `max_depth=5`, `random_state=42`
- Feature importance ranked by Gini impurity
- Tree structure visualised at depth ≤ 3

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `Sex_enc` | Label-encoded gender |
| `Embarked_enc` | Label-encoded embarkation port |
| `AgeGroup` | 0 = child, 1 = adult, 2 = elderly |
| `FamilySize` | SibSp + Parch + 1 |

Missing `Age` values are filled with the **median**; missing `Embarked` with the **mode**.

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run EDA
python eda.py

# 4. Train models and evaluate
python titanic_prediction.py
```

Outputs (charts) are saved to the `outputs/` folder.

## 📦 Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

## 📈 Results

| Model | Accuracy | ROC-AUC | CV Accuracy |
|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.86 | ~0.80 |
| Decision Tree | ~0.79 | ~0.82 | ~0.78 |

## 👤 Author

Your Name — feel free to fork and improve!

---
*Dataset source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)*
