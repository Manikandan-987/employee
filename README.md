# 🚢 Titanic Dataset - Survival Prediction Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)](https://www.python.org/)
[![Kaggle Ready](https://img.shields.io/badge/Kaggle-Compatible-blue)]()

This project uses the famous Titanic dataset to explore data analysis, feature engineering, and machine learning to predict which passengers survived the Titanic shipwreck.

---

## 📁 Dataset: `titanic.csv`

### 🔍 Overview

The Titanic dataset provides information on the passengers aboard the RMS Titanic. The goal is to predict whether a passenger survived or not using features like age, gender, ticket class, fare, and more.

---

### 🧾 Column Descriptions

| Column Name   | Description |
|---------------|-------------|
| `PassengerId` | Unique identifier assigned to each passenger |
| `Survived`    | Survival status (0 = No, 1 = Yes) |
| `Pclass`      | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name`        | Full name of the passenger |
| `Sex`         | Gender of the passenger |
| `Age`         | Age in years |
| `SibSp`       | Number of siblings/spouses aboard |
| `Parch`       | Number of parents/children aboard |
| `Ticket`      | Ticket number |
| `Fare`        | Fare paid for the ticket |
| `Cabin`       | Cabin number (if available) |
| `Embarked`    | Port of Embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) |

---

## 💡 Use Cases

- 🎯 Binary classification (survived vs. not survived)
- 📊 Exploratory data analysis and visualizations
- 🧠 Feature engineering and preprocessing practice
- 🤖 Supervised machine learning (Logistic Regression, Decision Trees, etc.)

---

## 🛠️ Technologies Used

- Python 🐍
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebooks

---

## 📈 Sample Python Code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('titanic.csv')

# View basic info
print(df.info())

# Visualize survival by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

# Check correlation between numerical features
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
