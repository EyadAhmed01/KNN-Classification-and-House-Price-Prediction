# KNN Classification and House Price Prediction

A machine learning project implementing K-Nearest Neighbors (KNN) classification and linear regression models for house price prediction. This project demonstrates both manual implementations from scratch and comparisons with scikit-learn's built-in algorithms.

## Project Overview

This project consists of two main components:

1. **KNN Classification**: Classifying telescope data using K-Nearest Neighbors algorithm
2. **House Price Prediction**: Predicting California house prices using various linear regression techniques

## Features

### KNN Classification
- Manual KNN implementation from scratch
- Comparison with scikit-learn's `KNeighborsClassifier`
- Hyperparameter tuning to find optimal k value
- Data preprocessing including class balancing and feature scaling
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)

### House Price Prediction
- Manual Linear Regression implementation using normal equation
- Gradient Descent implementation
- Comparison with scikit-learn models:
  - `LinearRegression`
  - `Ridge` Regression
  - `Lasso` Regression
- Proper data splitting to prevent data leakage
- Feature scaling using StandardScaler
- Evaluation metrics (MSE, R², MAE)

## Dataset

### Telescope Data
- **Location**: `telescope_data/telescope_data.csv`
- **Purpose**: Binary classification (classes: 'g' and 'h')
- **Preprocessing**: Class balancing and min-max scaling

### California Houses Data
- **Location**: `California_Houses/California_Houses.csv`
- **Purpose**: Regression (predicting `Median_House_Value`)
- **Preprocessing**: StandardScaler normalization

## Requirements

The project uses the following Python libraries:

- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning algorithms and utilities
  - `StandardScaler` - Feature scaling
  - `KNeighborsClassifier` - KNN classification
  - `LinearRegression`, `Ridge`, `Lasso` - Regression models
  - `train_test_split` - Data splitting
  - Various metrics for evaluation

## Project Structure

```
.
├── Machine_Learnig_Lab1.ipynb    # Main Jupyter notebook with all implementations
├── telescope_data/
│   └── telescope_data.csv         # Telescope classification dataset
└── California_Houses/
    └── California_Houses.csv      # California housing dataset
```

## Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Machine_Learnig_Lab1.ipynb
   ```

2. **Run all cells** to execute the complete workflow:
   - Data loading and preprocessing
   - Model training and evaluation
   - Comparison between manual and sklearn implementations

## Key Implementations

### KNN Classification
- **Distance Calculation**: Euclidean distance between validation and training points
- **K-Nearest Neighbors**: Finding k closest neighbors
- **Prediction**: Majority voting among k nearest neighbors
- **Hyperparameter Tuning**: Testing k values from 1 to 20 to find optimal performance

### Linear Regression
- **Normal Equation**: Manual implementation using matrix operations
- **Gradient Descent**: Iterative optimization algorithm
- **Regularization**: Ridge and Lasso regression for handling overfitting

## Results

The notebook includes:
- Validation accuracy plots for different k values
- Performance comparisons between manual and sklearn implementations
- Detailed evaluation metrics for both classification and regression tasks

## Notes

- Data splitting follows a 70% training, 15% validation, 15% test split
- Feature scaling is applied only to training data to prevent data leakage
- Target variables are preserved in their original scale for interpretability

## Author

Machine Learning Lab 1 Project

## License

This project is for educational purposes.
