# Developing Four Core Binary Classifiers from Scratch for Loan Approval Prediction and Assessing Dimensionality Reduction with PCA 

**Project Summary**  
This project focuses on evaluating the performance of four binary classifiers on real-world loan approval data and analyzing how Principal Component Analysis (PCA) affects their accuracy. The classifiers include:

- **Decision Tree**  
- **k-Nearest Neighbors (kNN)**  
- **Linear Discriminant Analysis (LDA)**  
- **Support Vector Machine (SVM)**  

Each classifier is applied to a dataset of 27 features, with a binary label indicating loan approval (0 for approved, 1 for denied). We examine the performance of these models by measuring Type 1 (false positive) and Type 2 (false negative) error rates. All classifiers are implemented from scratch to enhance understanding and control over the algorithms.

**Objectives**  
**Binary Classification Performance**  
- Evaluate each classifier's effectiveness using the original 27 features.  
- Measure Type 1 and Type 2 error rates on the test data.  

**Dimensionality Reduction with PCA**  
- Assess the impact of feature reduction on kNN and SVM classifiers.  
- Experiment with different numbers of principal components (K = 5, 10, 15).  
- Compare performance metrics between classifiers trained on original vs. PCA-reduced features.  

**Data Overview**  
- **Training Data:** 900 samples (450 approved, 450 denied)  
- **Testing Data:** 400 samples (200 approved, 200 denied)  
- **Features:** 27 attributes (e.g., Age, Annual Income)  

**Project Breakdown**  
**Classifiers with Original Features**  
- **LDA:** Use projections onto a direction vector (w) for classification.  
- **Decision Tree:** Build using criteria like Gini impurity or information gain.  
- **kNN:** Evaluate with (k = 1, 3, 5, and 10).  
- **SVM:** Apply soft-margin SVM to manage non-separable data.  

**Classifiers with PCA-Reduced Features**  
- Reduce the feature set using PCA and apply kNN and SVM with (K = 5, 10, and 15) components.  
- Compare the error rates to those from classifiers trained on original features.  

**Evaluation Metrics**  
- **Type 1 Error Rate:** Percentage of approved loans misclassified as denied.  
- **Type 2 Error Rate:** Percentage of denied loans misclassified as approved.  

**Analysis Focus**  
- How error rates differ among classifiers.  
- PCA's effect on classifier performance and optimal component selection.  

This analysis helps demonstrate the trade-offs and benefits of dimensionality reduction for binary classification, specifically for loan approval prediction.
