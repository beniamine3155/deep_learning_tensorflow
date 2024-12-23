# Experiment 2: Hyperparameter Tuning and Model Evaluation

## Overview

In this experiment, we performed hyperparameter tuning on selected machine learning models to optimize their performance. The process involved using grid search and cross-validation techniques to identify the best parameters for each model. The primary objective was to minimize the mean squared error (MSE) and improve prediction accuracy.

---

## Objective

To enhance the predictive capabilities of the machine learning models by identifying optimal hyperparameter configurations through systematic tuning.

---

## Dataset

- **Source:** [https://www.kaggle.com/c/house-prices-advanced-regression-techniques.]
- **Description:** The dataset contains features and target variables for [Problem Domain].
- **Preprocessing Steps:**
  - Handled missing values using the mean for numerical columns.
  - Encoded categorical variables using `OneHotEncoder`.
  - Split the dataset into training and testing sets (80%-20% split).

---

## Models Used

1. **Support Vector Machine (SVM)**
2. **Decision Tree**

---

## Methodology

### Hyperparameter Tuning

- Used **GridSearchCV** for systematic tuning.
- Performed cross-validation with `cv=3`.
- Evaluated performance using the **negative mean squared error** as the scoring metric.

### Parameter Grids

1. **Support Vector Machine (SVM):**

   - `C`: [0.1, 1, 10]
   - `kernel`: ["linear", "rbf"]

2. **Decision Tree:**
   - `max_depth`: [10, 20, 30]
   - `min_samples_split`: [2, 10, 20]

---

## Implementation

### Libraries Used

```python
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
```

### Code Snippets

#### Grid Search Implementation

```python
param_grids = {
    "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Decision Tree": {"max_depth": [10, 20, 30], "min_samples_split": [2, 10, 20]}
}

best_params = {}
for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    if model_name in param_grids:
        grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        models[model_name] = grid.best_estimator_
        best_params[model_name] = grid.best_params_
        print(f"Best parameters for {model_name}: {grid.best_params_}")
    else:
        model.fit(X_train, y_train)
```

---

## Results

### Best Parameters

1. **Support Vector Machine (SVM):**

   - `C`: [Best Value]
   - `kernel`: [Best Value]

2. **Decision Tree:**
   - `max_depth`: [Best Value]
   - `min_samples_split`: [Best Value]

### Evaluation Metrics

| Model                  | Best Parameters                              | Mean Squared Error (MSE) |
| ---------------------- | -------------------------------------------- | ------------------------ |
| Support Vector Machine | {"C": ..., "kernel": ...}                    | ...                      |
| Decision Tree          | {"max_depth": ..., "min_samples_split": ...} | ...                      |

---

## Challenges

- **Computational Time:** Grid search with large parameter grids was time-consuming.
- **Dataset Imbalance:** Addressing class or target imbalances required careful preprocessing.

---

## Lessons Learned

- Reducing the parameter grid and using `RandomizedSearchCV` can save computation time.
- Parallelizing grid search using `n_jobs=-1` accelerates the process.

---

## Future Improvements

- Explore advanced tuning techniques like **Bayesian Optimization** or **Optuna**.
- Investigate feature engineering methods to improve model performance.
- Experiment with additional models, such as Gradient Boosting or Neural Networks.

---

## References

- [GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Dataset Source](#)

---
