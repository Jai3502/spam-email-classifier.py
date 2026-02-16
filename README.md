# spam-email-classifier.py
A Python script that performs end-to-end Spam Email Classification using NLP techniques. It includes data cleaning, exploratory data analysis (EDA), text preprocessing, feature extraction with TF-IDF, and multiple machine learning models to identify spam messages with high accuracy and precision.

# Spam Email Classifier using Machine Learning

This project implements a complete **Spam Email Detection System** using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to classify messages as **Spam** or **Ham (Not Spam)** with high precision.

---

##  Project Overview

Spam messages are a common problem in digital communication. This project analyzes text data and builds multiple machine learning models to accurately detect spam messages.

The workflow includes:
- Data cleaning
- Exploratory Data Analysis (EDA)
- Text preprocessing
- Feature extraction (TF-IDF)
- Model training and evaluation
- Ensemble learning techniques

---

##  Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib & Seaborn**
- **NLTK**
- **Scikit-learn**
- **XGBoost**
- **WordCloud**

---

##  Dataset

- SMS Spam Collection Dataset  
- Contains labeled messages: `spam` and `ham`
- Dataset is preprocessed to remove duplicates and unnecessary columns

---

##  Project Workflow

### 1️ Data Cleaning
- Removed irrelevant columns
- Renamed columns for clarity
- Encoded target labels (spam = 1, ham = 0)
- Removed duplicate records

### 2️ Exploratory Data Analysis (EDA)
- Class distribution visualization
- Character, word, and sentence analysis
- Correlation heatmaps
- Spam vs Ham comparisons

### 3️ Text Preprocessing
- Converted text to lowercase
- Tokenization
- Removed special characters, stopwords, and punctuation
- Applied stemming using Porter Stemmer

### 4️ Feature Engineering
- TF-IDF Vectorization
- Converted text data into numerical form

### 5 Model Building
The following models were trained and evaluated:
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors
- Random Forest
- AdaBoost
- Gradient Boosting
- Extra Trees
- XGBoost

### 6️ Ensemble Techniques
- Voting Classifier
- Stacking Classifier

Models were evaluated using:
- Accuracy Score
- Precision Score
- Confusion Matrix

---

##  Results

- **Multinomial Naive Bayes with TF-IDF** performed exceptionally well
- Ensemble models further improved precision
- High spam detection accuracy with minimal false positives

---

##  How to Run the Project

1. Clone the repository
   ```bashh
   git clone https://github.com/your-username/spam-email-classifier.git
   
2. Install required libraries
   ````bash
   pip install numpy pandas matplotlib seaborn nltk scikit-learn xgboost wordcloud
   
3. Download NLTK resources
   ````bash
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   
4. Run the script
  ````bash
  python spam_email_classifier.py

git clone https://github.com/your-username/spam-email-classifier.git

