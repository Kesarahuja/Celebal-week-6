# Celebal-week-6
# Model Evaluation and Hyperparameter Tuning

This assignment demonstrates how to evaluate multiple machine learning models using standard classification metrics and apply hyperparameter tuning techniques to optimize performance.

## Assignment Objective

Train multiple machine learning models and evaluate their performance using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score

Use hyperparameter tuning techniques such as:

- GridSearchCV
- RandomizedSearchCV

Analyze the results to select the best-performing model.

---

## Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## Evaluation Metrics

The models were evaluated on the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score

These were computed using the Scikit-learn metrics module.

---

## Hyperparameter Tuning

Performed tuning using:

- GridSearchCV (Exhaustive parameter grid search)
- RandomizedSearchCV (Random parameter combinations)

Example tuned model:
- Support Vector Machine (SVM): optimized over kernel, C, and gamma.

---

## Results Summary

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9737   | 0.9722    | 0.9859 | 0.9790   |
| Decision Tree       | 0.9386   | 0.9444    | 0.9577 | 0.9510   |
| Random Forest       | 0.9649   | 0.9589    | 0.9859 | 0.9722   |
| SVM                 | 0.9825   | 0.9726    | 1.0000 | 0.9861   |
| KNN                 | 0.9474   | 0.9577    | 0.9577 | 0.9577   |

The SVM model was selected as the best performer due to its high F1-score and perfect recall.

---

## References

- KDNuggets Hyperparameter Tuning Guide:  
  https://www.kdnuggets.com/hyperparameter-tuning-gridsearchcv-and-randomizedsearchcv-explained
- Scikit-learn Documentation:  
  https://scikit-learn.org/stable/documentation.html

---
