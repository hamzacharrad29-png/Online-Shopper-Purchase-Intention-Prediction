# Project Title : Online Shoppers Purchase Intention Prediction

## Overview
This machine learning project predicts whether a website visitor will complete a purchase (Revenue = True) or leave without generating revenue.
The objective is to analyze customer browsing behavior and build predictive models that help businesses identify high-intent customers and improve conversion rates.

## Business Problem
E-commerce businesses receive thousands of visitors daily, but only a small percentage complete a purchase.
By predicting purchase intention, companies can:

- Target potential buyers
- Personalize marketing campaigns
- Improve customer experience
- Increase revenue
- Optimize advertising spending

## Dataset
Source: Kaggle - Online Shoppers Purchasing Intention Dataset

The dataset contains information about:

- Administrative pages visited
- Product-related pages
- Bounce rates
- Exit rates
- Visitor type
- Traffic source
- Month of visit
- Weekend activity

Target Variable:
- Revenue (Yes/No)

## Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Performance Comparison

## Exploratory Data Analysis

### Class Distribution

![Class Distribution](https://github.com/hamzacharrad29-png/Online-Shopper-Purchase-Intention-Prediction/blob/4dc824da378a23bf58b5cd890964f962ec19e346/Distribution%20des%20revenue%20True%20Vs%20False%20.png)

### Correlation Heatmap

![Correlation Heatmap](https://github.com/hamzacharrad29-png/Online-Shopper-Purchase-Intention-Prediction/blob/c7d024f3a0849c28472b22b9515f641466fb00b9/Correlation%20Heatmap%20.png)

## Machine Learning Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Bagging Classifier
- **Evaluate with Cross validation **

# Best Performing Model
**Decison Tree** achieved the highest performance and demonstrated strong generalization on unseen data.

![ROC Curve Model Comparaion](https://github.com/hamzacharrad29-png/Online-Shopper-Purchase-Intention-Prediction/blob/76baaf56fea6276ab07dc318320934296f274459/ROC%20Curve%20Model%20Comparaison.png)

![Confusion Matrix Best Model](https://github.com/hamzacharrad29-png/Online-Shopper-Purchase-Intention-Prediction/blob/16e02fc3bef27d6c1cfaaae24c4b0d0879e7cd77/confusion%20matrix%20best%20model%20.png)

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook
- Streamlit (to build the app)

## Web Application
To make the machine learning model accessible to non-technical users, I developed an interactive Streamlit application that predicts whether a website visitor is likely to complete a purchase.
## Features
- Interactive and user-friendly interface
- Real-time purchase intention prediction
- Probability score for each prediction
- Integration of the trained Decision Tree model (best model)

## User Inputs
The application allows users to enter all the details needed 
## Prediction Output
The application returns:
- Purchase Likelihood (Yes/No)
- Prediction Probability Score
## application image 
![application image](https://github.com/hamzacharrad29-png/Online-Shopper-Purchase-Intention-Prediction/blob/07359de798d7c720811ce329e4742edc041c6b77/APP%20.png)
## Business Value
This solution can help e-commerce teams identify high-intent visitors, improve customer targeting, optimize marketing strategies, and support data-driven decision-making.

## Author
Hamza Charrad

Aspiring Data Analyst | Machine Learning Enthusiast

LinkedIn:
[https://www.linkedin.com/in/hamza-charrad/]  

