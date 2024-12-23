# MNIST Digits Classification

This project involves classifying handwritten digits from the MNIST dataset. The goal is to use three machine learning algorithms (Logistic Regression, Support Vector Machine, and Decision Tree) to classify the digits and compare their performance.

## Steps

### 1. Dataset Download

- Download the MNIST Digits dataset from [Kaggle - Digit Recognizer](https://www.kaggle.com/c/digit-recognizer).

### 2. Load Data into Pandas DataFrame

- Read the dataset into a Pandas DataFrame for exploration and preprocessing.

### 3. Dataset Exploration

- Report the number of data points and features.
- Visualize sample images using Matplotlib.

### 4. Data Preprocessing

- Split the dataset into training and test sets.
- Normalize pixel values to the range [0, 1].
- Reshape the data as required.

### 5. Model Training

#### Algorithms:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Decision Tree**

#### Steps:

- Train each model on the dataset.
- Use hyperparameter tuning (e.g., GridSearchCV or RandomizedSearchCV) to select the best hyperparameters.
- Select the best model for each algorithm based on tuning results.

### 6. Model Evaluation

#### On Training Data:

- Generate confusion matrices.
- Compute accuracy, precision, recall, and F1 scores.

#### On Test Data:

- Generate confusion matrices.
- Compute accuracy, precision, recall, and F1 scores.
- Use bootstrapping to measure average accuracy, precision, recall, and their 95% confidence intervals.

### 7. Results Comparison

- Compare training and test results for each model.
- Identify the best model based on performance metrics.
- Analyze overfitting or underfitting for each model.

## Project Files

1. **Dataset**: The MNIST dataset in CSV format.
2. **Python Code**: A Python script implementing the above steps.
3. **Results**: Metrics and visualizations generated during evaluation.

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
