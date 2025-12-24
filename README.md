# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries
Create a sample email dataset
Convert labels to binary values
Split data into features and labels
Split into training and testing sets
Convert text data into numerical form using TF-IDF
Train SVM model
Make predictions
Evaluate the model
Test with a new email
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = {
    'message': [
        'Win money now',
        'Limited offer buy now',
        'Meeting scheduled tomorrow',
        'Your invoice is attached',
        'Congratulations you won a prize',
        'Let us catch up for lunch',
        'Free entry in a contest',
        'Project deadline is tomorrow',
        'Claim your free reward',
        'Can we discuss the report'
    ],
    'label': [
        'spam', 'spam', 'ham', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham'
    ]
}

df = pd.DataFrame(data)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)

y_pred = svm_model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_email = ["Congratulations! You have won a free gift"]
new_email_vec = vectorizer.transform(new_email)
prediction = svm_model.predict(new_email_vec)

if prediction[0] == 1:
    print("\nPrediction: Spam Mail")
else:
    print("\nPrediction: Not Spam Mail")

/*
Program to implement the SVM For Spam Mail Detection..
Developed by: dhanalakshmi.c
RegisterNumber: 25018616 
*/
```

## Output:
<img width="1280" height="703" alt="image" src="https://github.com/user-attachments/assets/73df3170-7cc7-4623-925a-3803b8bb5b21" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
