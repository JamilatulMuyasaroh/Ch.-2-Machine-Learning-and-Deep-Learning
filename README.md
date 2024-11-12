# Ch.-2-Machine-Learning-and-Deep-Learning
This repository includes a collection of practical assignments designed to develop skills in machine learning and deep learning techniques through **classification**, **clustering**, **regression**, and **fraud detection** tasks. Each task utilizes distinct algorithms and frameworks to address specific data challenges.

## Notebooks Overview

### Case 1: Customer Exit Prediction

- **Objective**: Develop a classification model to predict customer churn based on various banking data attributes.
- **Dataset**: `SC_HW1_bank_data.csv`
- **Libraries Used**: Pandas, NumPy, Scikit-learn

#### Steps:
1. **Data Preprocessing**: Clean the dataset by removing irrelevant columns, apply one-hot encoding for categorical variables, and normalize the data.
2. **Modeling**: Train three classification models—Random Forest, Support Vector Classifier (SVC), and Gradient Boosting—to predict churn likelihood.
3. **Hyperparameter Tuning**: Use grid search to optimize model parameters for the best results.
4. **Evaluation**: Evaluate model performance using accuracy scores, classification reports, and confusion matrices.

- **Outcome**: The Gradient Boosting model achieved the highest accuracy, making it the most effective choice for churn prediction with optimal processing time.

---

### Case 2: Customer Data Segmentation with KMeans Clustering

- **Objective**: Segment customer data into clusters using unsupervised learning.
- **Dataset**: `cluster_s1.csv`
- **Libraries Used**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

#### Steps:
1. **Data Preparation**: Remove unnecessary columns and preprocess the dataset.
2. **Optimal Cluster Count**: Determine the best number of clusters (k) using the Silhouette Score.
3. **Modeling with KMeans**: Cluster data into distinct segments and visualize the results.

- **Outcome**: The ideal cluster count was determined using the Silhouette Score, with data visualizations highlighting clear cluster boundaries.

---

### Case 3: California House Price Prediction using Neural Networks

- **Objective**: Predict California housing prices using a neural network.
- **Dataset**: California House Price dataset from Scikit-Learn
- **Libraries Used**: Pandas, NumPy, TensorFlow, Keras, Scikit-learn, Matplotlib

#### Steps:
1. **Data Preparation**: Split the data into training, validation, and test sets, followed by standardization and normalization.
2. **Model Building**: Create a Multilayer Perceptron (MLP) neural network with two input layers.
3. **Training and Evaluation**: Train the neural network model while monitoring loss to prevent overfitting.
4. **Model Saving**: Save the trained model for future predictions.

- **Outcome**: The neural network effectively predicts house prices, with metrics and visualizations indicating strong model performance.

---

### Case 4: Fraud Detection in Credit Card Transactions

- **Objective**: Build a classification model to identify fraudulent transactions.
- **Dataset**: `Credit Card Fraud 2023`
- **Libraries Used**: Pandas, cuDF, cuML, NumPy (cuPy), Scikit-learn, PyTorch

#### Steps:
1. **Data Loading with GPU**: Use GPU processing for efficient data handling and preprocessing.
2. **Data Conversion**: Convert data into tensors and prepare it for PyTorch DataLoader.
3. **Model Building**: Design a multilayer perceptron with four hidden layers in PyTorch.
4. **Training and Evaluation**: Train and fine-tune the model to achieve a minimum accuracy of 95%.

- **Outcome**: The model successfully achieves high accuracy, proving effective for fraud detection in real-time credit card transactions.

---

## Running the Notebooks

To run each notebook:

1. Open **Google Colab**.
2. Upload the notebook files (`02_Kelompok_G_1.ipynb` to `02_Kelompok_G_4.ipynb`).
3. Execute each cell in sequence, following the provided instructions within each notebook for a smooth setup and analysis.

This repository is ideal for hands-on practice with **machine learning** and **deep learning** in real-world scenarios, covering essential techniques like **classification**, **clustering**, **regression**, and **fraud detection**.
