# -*- coding: utf-8 -*-
"""Spam Mail Prediction SVM & LR.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zLyDCel_02ZAbqjEqLAigxkPjWb5uDGI
"""



"""**SPAM MAIL PREDICTION USING BINARY CLASSIFICATION (svm and logistic Regression)**"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# DATA COLLECTION AND DATA PREPROCESSING

mix_mail_data = pd.read_csv ('/content/mail_data.csv')

print (mix_mail_data)

mix_mail_data.head()

mix_mail_data.isnull().values.any()

mix_mail_data.isnull().sum ()

# replace the null values with a null string (incase of null values )
mail_data = mix_mail_data.where((pd.notnull(mix_mail_data)),'')

mail_data.shape

# Label Encoding

# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1



"""SPAM = 0
HAM = 1
"""

# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']

print (X)
print (Y)



"""*Splitting the Dataset into training Dataset and test dataset *"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)



"""# Feature Extraction: Transform the text data to feature vectors using TF-IDF"""

feature_extraction = TfidfVectorizer(min_df=3, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

"""# TRAINING AND EVALUATION"""

# Train and evaluate SVM model
svm_model = LinearSVC()
svm_model.fit(X_train_features, Y_train)



"""# Prediction on training data"""

svm_train_predictions = svm_model.predict(X_train_features)
svm_train_accuracy = accuracy_score(Y_train, svm_train_predictions)

svm_train_accuracy

# Prediction on test data
svm_test_predictions = svm_model.predict(X_test_features)
svm_test_accuracy = accuracy_score(Y_test, svm_test_predictions)

svm_test_accuracy



"""LOGISTIC REGRESSION"""

# Train and evaluate Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_features, Y_train)

# Prediction on training data
lr_train_predictions = lr_model.predict(X_train_features)
lr_train_accuracy = accuracy_score(Y_train, lr_train_predictions)

lr_train_accuracy

# Prediction on test data
lr_test_predictions = lr_model.predict(X_test_features)
lr_test_accuracy = accuracy_score(Y_test, lr_test_predictions)

lr_test_accuracy

# PREDICTIVE SYSTEM

input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have been wonderful and a blessing at all times"]

# Convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# Making prediction using SVM
svm_prediction = svm_model.predict(input_data_features)
print(f"SVM Prediction: {'Ham mail' if svm_prediction[0] == 1 else 'Spam mail'}")

# Making prediction using Logistic Regression
lr_prediction = lr_model.predict(input_data_features)
print(f"Logistic Regression Prediction: {'Ham mail' if lr_prediction[0] == 1 else 'Spam mail'}")

models = pd.DataFrame({
    "Model": ["SVM", "Logistic Regression"],
    " Test Accuracy Score": [svm_test_accuracy, lr_test_accuracy] ,
    " Train Accuracy SCore" : [svm_train_accuracy , lr_train_accuracy]
})

models

import seaborn as sns
sns.barplot(x=" Test Accuracy Score", y="Model",data=models)
models.sort_values(by=" Test Accuracy Score", ascending = False)

import seaborn as sns
sns.barplot(x=" Train Accuracy SCore", y="Model",data=models)
models.sort_values(by=" Test Accuracy Score", ascending = False)

