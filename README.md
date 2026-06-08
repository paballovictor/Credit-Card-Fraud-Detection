# Credit Card Fraud Detection with Imbalanced Datasets

## Project Overview

This project investigates how machine learning models can be used to detect fraudulent credit card transactions in a highly imbalanced dataset. The dataset contains mostly normal transactions, with fraud cases representing only about **0.17%** of all transactions.

Because of this strong imbalance, a model trained directly on the original dataset may achieve high accuracy by simply predicting most transactions as non-fraud. However, this would be misleading because the main goal is to correctly identify fraudulent transactions while also avoiding too many false alarms on normal transactions.

The notebook explores different strategies for dealing with class imbalance, including:

- Random undersampling
- Outlier removal
- Logistic regression and other classifiers
- SMOTE oversampling
- Neural network models
- Confusion matrix and precision-recall evaluation

---

## Dataset Description

The dataset contains anonymized credit card transaction data. Due to privacy reasons, most features are labelled as `V1`, `V2`, ..., `V28`.

The main columns are:

- `Time`: Time elapsed between transactions
- `Amount`: Transaction amount
- `V1` to `V28`: PCA-transformed features
- `Class`: Target variable  
  - `0` = Non-fraud transaction  
  - `1` = Fraud transaction  

The dataset has no missing values.

The class distribution is highly imbalanced:

| Class | Description | Proportion |
|---|---|---|
| 0 | Non-fraud | 99.83% |
| 1 | Fraud | 0.17% |

There are **492 fraud cases** in the dataset.

---

## Project Goals

The main goals of the project are:

1. Understand the structure and distribution of the credit card fraud dataset.
2. Show why class imbalance is a serious problem in fraud detection.
3. Create a balanced dataset using random undersampling.
4. Study feature correlations after balancing the dataset.
5. Remove extreme outliers from highly correlated features.
6. Train and compare multiple classification models.
7. Use SMOTE oversampling to improve fraud detection.
8. Compare traditional machine learning models with neural network models.
9. Evaluate models using metrics that are more useful than accuracy alone.

---

## Data Preprocessing

The `Time` and `Amount` columns were scaled so that they are consistent with the PCA-transformed features.

A balanced subsample was then created using:

- 492 fraud transactions
- 492 non-fraud transactions

This produced a 50/50 class distribution, allowing the models to better learn fraud-related patterns.

---

## Exploratory Data Analysis

The notebook first shows that the original dataset is highly imbalanced. This is important because using the original dataset directly can lead to models that are biased toward predicting non-fraud.

A balanced subsample was used to study correlations between features and the fraud label.

### Important Negative Correlations

The following features were negatively correlated with fraud:

- `V17`
- `V14`
- `V12`
- `V10`

This means that lower values of these features are associated with a higher probability of fraud.

### Important Positive Correlations

The following features were positively correlated with fraud:

- `V2`
- `V4`
- `V11`
- `V19`

This means that higher values of these features are associated with a higher probability of fraud.

---

## Outlier Removal

The notebook applies anomaly detection using the Interquartile Range method.

Extreme outliers were removed from highly correlated fraud-related features, especially:

- `V14`
- `V12`
- `V10`

After outlier removal, the balanced dataset was reduced from 984 rows to **947 rows**.

This step was used to reduce noise and improve model performance.

---

## Models Used

The notebook compares several classification models:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier
- Decision Tree Classifier
- Neural Network

The main imbalance-handling methods used were:

- Random undersampling
- SMOTE oversampling

---

## Random Undersampling Results

Random undersampling was used to create a balanced dataset by reducing the number of non-fraud transactions.

The tested classifiers performed well on the undersampled test set.

| Model | Accuracy | Fraud Recall | Fraud F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.94 | 0.90 | 0.94 |
| K-Nearest Neighbors | 0.93 | 0.86 | 0.92 |
| Support Vector Classifier | 0.93 | 0.88 | 0.93 |
| Decision Tree | 0.93 | 0.87 | 0.92 |

Logistic Regression gave the best overall performance on the undersampled dataset.

However, undersampling removes many non-fraud transactions, which means useful information may be lost.

---

## SMOTE Oversampling Results

SMOTE was used to balance the training data by creating synthetic fraud examples rather than removing non-fraud examples.

This allowed the model to keep more information from the original dataset.

The Logistic Regression model trained with SMOTE achieved:

| Metric | Value |
|---|---:|
| Accuracy | 0.987 |
| Average precision-recall score | 0.75 |
| Fraud precision | 0.10 |
| Fraud recall | 0.86 |
| Fraud F1-score | 0.19 |

The SMOTE model had much better precision-recall performance than undersampling.

The undersampling precision-recall score was only:

| Method | Average Precision-Recall Score |
|---|---:|
| Random Undersampling | 0.03 |
| SMOTE Oversampling | 0.75 |

This shows that SMOTE was much more effective when evaluated on the original imbalanced test set.

---

## Final Logistic Regression Comparison

The notebook compares the final Logistic Regression scores for undersampling and SMOTE.

| Technique | Score |
|---|---:|
| Random Undersampling | 0.942 |
| Oversampling with SMOTE | 0.987 |

Although SMOTE achieved a higher score, the notebook also notes that accuracy can be misleading in imbalanced classification problems. Precision, recall, F1-score, confusion matrices, and precision-recall curves are more meaningful for fraud detection.

---

## Neural Network Results

A simple neural network was also tested using both random undersampling and SMOTE.

The neural network architecture included:

- Input layer with 30 features
- One hidden layer with 32 nodes
- Output layer with 2 classes
- Softmax activation for classification

### Random Undersampling Neural Network

Confusion matrix on the original test set:

| Actual / Predicted | No Fraud | Fraud |
|---|---:|---:|
| No Fraud | 55,148 | 1,715 |
| Fraud | 8 | 90 |

This model detected most fraud cases, but it also incorrectly classified many normal transactions as fraud.

### SMOTE Neural Network

Confusion matrix on the original test set:

| Actual / Predicted | No Fraud | Fraud |
|---|---:|---:|
| No Fraud | 56,851 | 12 |
| Fraud | 33 | 65 |

The SMOTE neural network made far fewer false fraud predictions, but it missed more fraud cases compared to the undersampling neural network.

---

## Main Findings

The main findings from the notebook are:

1. The dataset is extremely imbalanced, with fraud making up only about 0.17% of transactions.
2. Accuracy alone is not a reliable metric for fraud detection.
3. Random undersampling helps models learn fraud patterns but removes many useful non-fraud examples.
4. SMOTE preserves more information by generating synthetic fraud examples.
5. Logistic Regression performed strongly compared with other classical classifiers.
6. SMOTE gave a much better precision-recall score than random undersampling.
7. Neural networks showed different trade-offs:
   - Undersampling detected more fraud cases but created many false positives.
   - SMOTE reduced false positives but missed more fraud cases.
8. The best model depends on the business goal:
   - If detecting as many fraud cases as possible is the priority, higher recall is preferred.
   - If reducing false alarms is important, precision and false-positive rate must also be considered.

---

## Conclusion

This project demonstrates the importance of handling imbalanced datasets carefully in fraud detection problems. A model can appear highly accurate while still performing poorly on the minority class.

The notebook shows that SMOTE oversampling improves performance compared with random undersampling, especially when evaluated using the precision-recall curve. However, there is still a trade-off between catching more fraud cases and avoiding false fraud alerts.

Future improvements could include:

- Applying outlier removal to the SMOTE dataset
- Testing more advanced ensemble models
- Tuning decision thresholds
- Comparing cost-sensitive learning approaches
- Evaluating models using business-cost metrics
- Using cross-validation pipelines to avoid data leakage

---

## Technologies Used

The notebook uses the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- TensorFlow / Keras

---

## How to Run the Notebook

1. Install the required Python libraries.
2. Load the credit card fraud dataset.
3. Run the notebook cells in order.
4. Review the exploratory analysis, model training, and evaluation results.

Required libraries include:

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow
keras
