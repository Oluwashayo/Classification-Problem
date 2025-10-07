
This project focuses on building and evaluating multiple machine learning models to detect whether an email is **spam** or **not spam**.  
The dataset was sourced from [Hugging Face](https://huggingface.co/datasets/UniqueData/email-spam-classification), and the project demonstrates how to preprocess text data, extract features using **TF-IDF**, train various classifiers, and compare their performances using **stratified cross-validation**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Techniques Used](#techniques-used)
4. [Models Implemented](#models-implemented)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Result](#result)

---

## Project Overview

Email spam detection is a **binary classification problem** where the goal is to identify emails that are unsolicited or malicious (spam) versus legitimate ones (ham).  
This project compares the performance of **five classical machine learning models** using **TF-IDF feature extraction**.

---

## Dataset Description

**Source:** [UniqueData/email-spam-classification](https://huggingface.co/datasets/UniqueData/email-spam-classification)

| Column | Description |
|---------|--------------|
| `text`  | The email message content |
| `label` | `1` for spam, `0` for not spam |

**Objective:** Predict whether an email is spam or not based on its content.

---

## Techniques Used

- **Natural Language Processing (NLP)** for text-based learning  
- **TF-IDF (Term Frequency–Inverse Document Frequency)** for feature extraction  
- **Stratified K-Fold Cross-Validation** to ensure balanced evaluation  
- **Model comparison** through visualization of accuracy scores  

---

## Models Implemented

| Model | Description | Key Advantage |
|--------|--------------|----------------|
| **Logistic Regression** | Learns a linear boundary between classes | Simple and efficient |
| **Multinomial Naive Bayes** | Probabilistic model assuming word independence | Excellent for text classification |
| **Linear SVC** | Maximizes class separation with support vectors | Works well with high-dimensional text data |
| **Random Forest** | Ensemble of decision trees | Reduces overfitting |
| **K-Nearest Neighbors (KNN)** | Classifies based on nearby samples | Simple, non-parametric method |

---

## Evaluation Metrics

The models were evaluated using:

- **Accuracy** – proportion of correct predictions  
- **Confusion Matrix** – detailed breakdown of true/false predictions  
- **Cross-validation Mean & Standard Deviation** – measures model stability  
- **ROC AUC** – evaluates how well the model distinguishes classes  

---
**Result:**, **KNN produced the highest test accuracy** in this experimental compared to other classifiers used in this setup.
