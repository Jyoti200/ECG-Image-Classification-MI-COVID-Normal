# SmartECG
A Comprehensive Approach to Automated Classification of Cardiac and COVID-19 Patients Using 12-Lead ECG Images – Integrating Machine Learning, Deep Neural Network, Convolutional Neural Network


## 📌 Overview
This project classifies ECG images into 5 categories: Myocardial Infarction (MI), COVID-19, History of Myocardial Infarction, Abnormal heartbeat and Normal using deep learning.

## 🚀 Features
- Uses a Convolutional Neural Network (CNN) for image classification
- Trained on the data [https://data.mendeley.com/datasets/gwbz3fsgp8/1]
- Flask web app for easy image upload and prediction

## 🔧 Installation & Setup
1. Clone the repo:
   ```sh
   git clone https://github.com/Jyoti200/ECG-Image-Classification-MI-COVID-Normal.git
   cd ECG-Image-Classification-MI-COVID-Normal

**Table of Contents**

1. Introduction
2. Dependencies


**Introduction**
SmartECG is designed to enhance the diagnostic process by leveraging state-of-the-art technologies in artificial intelligence. It focuses on automated classification of patients based on 12-lead ECG images, providing valuable insights into cardiac and COVID-19-related conditions.

**Machine** **Learning** **Algorithm**
Support Vector Machine was used to classify ECG images and got 85% accuracy. Random Forest got 83%  and Decision Tree to classify ECG images got 76% accuracy.

**CNN**
As Convolutional neural networks are more suitable for image classification, CNN architecture with a relu activation function was built, the optimizer was ADAM and the loss calculated was Sparse_categorical_crossentropy.
The model achieved 87% accuracy.

![image](https://github.com/Jyoti200/ECG_image_Classification_for_Myocardia_infarction_Covid-19_Normal/assets/86410759/3c3b09cf-5c39-4353-a525-d2c9671a37e1)


**Dependencies**
SmartECG relies on the following dependencies:

Python 3.11
TensorFlow
Scikit-learn
OpenCV

Ensure these dependencies are installed before running the application.









