# -*- coding: utf-8 -*-
"""Spam Mail Predicton.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1f0wOQAjCjnMLOV1VTOyNH0CmK17xBjSL
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# DATA COLLECTION AND DATA PREPROCESSING

mix_mail_data = pd.read_csv ('/content/mail_data.csv')

print (mix_mail_data)

mix_mail_data.head()

from matplotlib import pyplot as plt
import seaborn as sns
_df_1.groupby('Message').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

mix_mail_data.isnull().values.any()

mix_mail_data.isnull().sum ()

# replace the null values with a null string (incase of null values )
mail_data = mix_mail_data.where((pd.notnull(mix_mail_data)),'')

mix_mail_data.shape

mail_data.shape

# Label Encoding

# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1



"""SPAM == 0
HAM == 1

"""

# separating the data as texts and label

X = mail_data['Message']

Y = mail_data['Category']

print (X)
print (Y)



"""**Splitting the Dataset into training Dataset and test dataset **"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

print (X_train.shape)
print (X_test.shape)



"""Feature Extraction -



"""

# transforming the text data to feature vectors that can be used as input to the Logistic regression and for svm model

feature_extraction = TfidfVectorizer(min_df = 3, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

"""Training Model

"""

# training the model with svm

model1 = LinearSVC()
model1.fit (X_train_features , Y_train )

# evaluation and prediction

prediction_on_training_data = model1.predict (X_train_features)
accuracy_obtained = accuracy_score ( Y_train , prediction_on_training_data)

accuracy_obtained

# prediction on test data

prediction_on_test_data = model1.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

accuracy_on_test_data

"""predictive system"""

input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')