# 🐟 Deep CNN Model for Fish Growth Estimation 🌱

Welcome to the Deep CNN Model for Fish Growth Estimation from Larvae Statistics project! 🎉 This repository contains a machine learning solution to predict fish growth using larvae data, developed as a part of a Bachelor of Technology project at St. Martin's Engineering College. 📚

## 📖 Project Overview

This project leverages a 1D Convolutional Neural Network (CNN) and Support Vector Regressor (SVR) to estimate fish growth from larvae statistics, aiming to support sustainable aquaculture practices. The application features a user-friendly Tkinter GUI for seamless interaction, allowing users to upload datasets, preprocess data, train models, and make predictions. 🖥️

## 🔍 Problem Identified

Manual Measurement Challenges 📏: Traditional fish growth estimation relies on labor-intensive manual measurements, leading to human errors and data gaps.

Scalability Issues 📈: Conventional methods struggle to handle large datasets in large-scale aquaculture farms.

Environmental Variability 🌡️: Traditional models fail to account for real-time environmental factors like temperature and pH, resulting in inconsistent predictions.

Lack of Automation 🤖: The absence of automated systems increases dependency on human observation, hindering real-time decision-making.



## 🛠️ Problem Solved

Automation with ML 🧠: Implemented a 1D CNN and SVR to automate fish growth prediction, reducing manual effort.

Scalable Data Processing 📊: Used PCA for dimensionality reduction and StandardScaler for normalization, enabling efficient handling of large datasets.

Dynamic Feature Handling ⚙️: Preprocessed data by encoding categorical variables and handling missing values, ensuring robust predictions despite environmental variability.

User-Friendly Interface 🖱️: Developed a Tkinter-based GUI for easy dataset upload, preprocessing, model training, and prediction, making the tool accessible to non-technical users.

## 🏆 Achievements

High Accuracy 🎯: The CNN model achieved an impressive R² score of 0.99, with an MSE of 0.0000 and MAE of 0.0018, outperforming the SVR (MSE: 0.0010, MAE: 0.0246).

Efficient Preprocessing 🧹: Successfully handled missing data and encoded categorical features like cruise_id, brief_desc, and common_name using one-hot encoding.

Visual Insights 📉: Generated correlation heatmaps and scatter plots to visualize feature relationships and model performance.

Practical Application 🌍: Built a scalable tool for aquaculture farms, fish hatcheries, and fisheries research to optimize feeding cycles and support sustainable practices.

## 🚀 Features

📂 Dataset Upload: Upload fish larvae datasets in CSV format.

🧹 Preprocessing: Handle missing values, encode categorical features, and visualize correlations with heatmaps.

📈 Model Training: Train both SVR and 1D CNN models to predict fish growth.

📊 Performance Metrics: Evaluate models using MSE, MAE, and R² scores, with scatter plots for visualization.

🔮 Prediction: Make growth predictions on new datasets using the trained CNN model.


## 🛠️ Technologies Used

Python 🐍: Core programming language.

Tkinter 🖥️: For the GUI.

TensorFlow/Keras 🧠: For building and training the 1D CNN model.

Scikit-learn 📊: For SVR, PCA, and data preprocessing.

Pandas/NumPy 📚: For data manipulation.

Matplotlib/Seaborn 📉: For data visualization.



## 📋 Requirements

Software: Python 3.7+, Tkinter, TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.

Hardware: Minimum Intel i3 processor, 4GB RAM, 250GB hard disk.


## 📊 Results

SVR Model: MSE: 0.0010, MAE: 0.0246.

CNN Model: MSE: 0.0000, MAE: 0.0018, R²: 0.9900.

Visualizations include correlation heatmaps and scatter plots comparing predicted vs. actual growth values.


## 🌟 Future Enhancements

Integrate environmental factors (e.g., temperature, pH) and genetic data to improve predictions. 🌡️🧬

Explore ensemble methods for even better accuracy. 🤝

Add IoT integration for real-time data collection. 📡
