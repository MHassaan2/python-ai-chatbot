print('Phase 4 --> Intent Classification \n')
#import libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
# Display all columns
pd.set_option('display.max_columns', None)
print("Libraries imported successfully!")

def preprocess_sample_text(text):
    text = str(text).lower()
    text = ''.join(
        char if char.isalpha() or char.isspace() else ' '
        for char in text
    )
    return " ".join(text.split())

#load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(script_dir, "..", "data")
df = pd.read_csv(os.path.join(data_dir, "final_processed_dataset.csv"))
print("Dataset loaded successfully!")
print(df.head())
print(df.columns)

#check required columns
required_columns = ['processed_text', 'intent']
for column in required_columns:
    if column in df.columns:
        print(f"{column} column found!")
    else:
        print(f"{column} column NOT found!")

#select input and output
X = df['processed_text']
y = df['intent']
print("Input and output selected successfully!")

#label encoder
intent_encoder = LabelEncoder()
y_encoded = intent_encoder.fit_transform(y)
print("Labels encoded successfully!")
print("\nEncoded Classes:\n")
for i, label in enumerate(intent_encoder.classes_):
    print(f"{label} --> {i}")

#load TF-IDF vectorizer
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "tfidf_vectorizer.pkl")
with open(model_path, "rb") as file:
    vectorizer = pickle.load(file)
print("TF-IDF vectorizer loaded successfully!")

#text transformation into TF-IDF
X_tfidf = vectorizer.transform(X)
print("TF-IDF transformation completed!")
print("\nTF-IDF Shape:")
print(X_tfidf.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y_encoded,
    test_size=0.4,
    random_state=42,
    stratify=y_encoded
)
print("Train-test split completed!")
print("\nTraining Shape:")
print(X_train.shape)
print("\nTesting Shape:")
print(X_test.shape)

#logistic regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
print("Logistic Regression trained successfully!")
#logistic regression prediction
lr_predictions = lr_model.predict(X_test)
#logistic regression accuracy
lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)
print("Logistic Regression Accuracy:")
print(lr_accuracy)

#logistic regression classification report
print("\n===================================================")
print("INTENT LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("===================================================\n")
print(
    classification_report(
        y_test,
        lr_predictions
    )
)

#logistic regression confusion matrix
lr_cm = confusion_matrix(
    y_test,
    lr_predictions
)
plt.figure(figsize=(10, 7))
sns.heatmap(
    lr_cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.close()


#naive bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("\nNaive Bayes trained successfully!")
#naive bayes prediction
nb_predictions = nb_model.predict(X_test)
print("\nNaive Bayes predictions completed!")
#naive bayes accuracy
nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)
print("Naive Bayes Accuracy:")
print(nb_accuracy)

#naive bayes classification report
print("\n===================================================")
print("NAIVE BAYES CLASSIFICATION REPORT")
print("===================================================\n")
print(
    classification_report(
        y_test,
        nb_predictions
    )
)

#naive bayes confusion matrix
nb_cm = confusion_matrix(
    y_test,
    nb_predictions
)
plt.figure(figsize=(10, 7))
sns.heatmap(
    nb_cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.close()

#svm model
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
print("\nSVM model trained successfully!")
#svm model prediction
svm_predictions = svm_model.predict(X_test)
print("\nSVM predictions completed!")
#svm model accuracy
svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)
print("SVM Accuracy:")
print(svm_accuracy)

#svm model classification report
print("\n===================================================")
print("SVM CLASSIFICATION REPORT")
print("===================================================\n")
print(
    classification_report(
        y_test,
        svm_predictions
    )
)

#svm model confusion matrix
svm_cm = confusion_matrix(
    y_test,
    svm_predictions
)
plt.figure(figsize=(10, 7))
sns.heatmap(
    svm_cm,
    annot=True,
    fmt='d',
    cmap='Oranges'
)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.close()

#model comparison
models = ['Logistic Regression', 'Naive Bayes', 'SVM']
accuracies = [
    lr_accuracy,
    nb_accuracy,
    svm_accuracy
]
comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies
})
print(comparison_df)

#visualization
plt.figure(figsize=(8, 5))
sns.barplot(
    x='Model',
    y='Accuracy',
    data=comparison_df
)
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.close()

#best model selection
best_accuracy = max(accuracies)
best_model_name = models[
    accuracies.index(best_accuracy)
]
print("\nBest Performing Model:")
print(best_model_name)
print("\nBest Accuracy:")
print(best_accuracy)

#select best model
trained_models = {
    'Logistic Regression': lr_model,
    'Naive Bayes': nb_model,
    'SVM': svm_model
}
best_model = trained_models[best_model_name]
model_dir = os.path.join(project_dir, "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "intent_model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(best_model, file)
print("Intent classification model saved successfully!")
#save label encoder
encoder_path = os.path.join(model_dir, "intent_encoder.pkl")
with open(encoder_path, "wb") as file:
    pickle.dump(intent_encoder, file)
print("Intent label encoder saved successfully!")
#custom prediction
sample_text = "I feel very stressed and anxious"
# Preprocess text
sample_vector = vectorizer.transform([preprocess_sample_text(sample_text)])
# Predict
prediction = best_model.predict(sample_vector)
# Decode label
predicted_intent = intent_encoder.inverse_transform(
    prediction
)
print("\nPredicted Intent:")
print(predicted_intent[0])

#multiple sample prediction
sample_inputs = [
    "I feel very lonely",
    "I am extremely anxious today",
    "Nobody understands me",
    "I feel stressed because of studies",
    "I am feeling hopeless"
]
print("\n===================================================")
print("MULTIPLE SAMPLE PREDICTIONS")
print("===================================================\n")
for text in sample_inputs:
    vector = vectorizer.transform([preprocess_sample_text(text)])
    prediction = best_model.predict(vector)
    decoded_prediction = intent_encoder.inverse_transform(
        prediction
    )
    print(f"Input: {text}")
    print(f"Predicted Intent: {decoded_prediction[0]}")
    print("------------------------------------------------")

#final summary
print("\n===================================================")
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("===================================================")

print(f"\nFinal Dataset Shape: {df.shape}")

print("\nTasks Completed:")
print("1. Dataset Loaded")
print("2. Labels Encoded")
print("3. TF-IDF Features Loaded")
print("4. Train-Test Split Completed")
print("5. Logistic Regression Trained")
print("6. Naive Bayes Trained")
print("7. SVM Trained")
print("8. Model Comparison Completed")
print("9. Confusion Matrix Created")
print("10. Best Model Saved")
print("11. Label Encoder Saved")
print("12. Custom Predictions Tested")
print ('Phase 5 --> Emotion Classification\n')
#check required columns
required_columns = ['processed_text', 'emotion']
for column in required_columns:
    if column in df.columns:
        print(f"{column} column found!")
    else:
        print(f"{column} column NOT found!")

#select input and output
X = df['processed_text']
y = df['emotion']
print("Input and output selected successfully!")

#label encoder
emotion_encoder = LabelEncoder()
y_encoded = emotion_encoder.fit_transform(y)
print("Labels encoded successfully!")
print("\nEncoded Classes:\n")
for i, label in enumerate(emotion_encoder.classes_):
    print(f"{label} --> {i}")

#load TF-IDF vectorizer
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "models")
with open(os.path.join(data_dir, "tfidf_vectorizer.pkl"), 'rb') as file:
    vectorizer = pickle.load(file)
print("TF-IDF vectorizer loaded successfully!")

#text transformation into TF-IDF
X_tfidf = vectorizer.transform(X)
print("TF-IDF transformation completed!")
print("\nTF-IDF Shape:")
print(X_tfidf.shape)

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y_encoded,
    test_size=0.4,
    random_state=42,
    stratify=y_encoded
)
print("Train-test split completed!")
print("\nTraining Shape:")
print(X_train.shape)
print("\nTesting Shape:")
print(X_test.shape)

#logistic regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
print("Logistic Regression trained successfully!")
#logistic regression prediction
lr_predictions = lr_model.predict(X_test)
#logistic regression accuracy
lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)
print("Logistic Regression Accuracy:")
print(lr_accuracy)
print("\n===================================================")
print("EMOTION LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("===================================================\n")
#logistic regression classification report
print(
    classification_report(
        y_test,
        lr_predictions
    )
)

#logistic regression confusion matrix
lr_cm = confusion_matrix(
    y_test,
    lr_predictions
)
plt.figure(figsize=(10, 7))
sns.heatmap(
    lr_cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.close()

#naive bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("\nNaive Bayes trained successfully!")
#naive bayes prediction
nb_predictions = nb_model.predict(X_test)
print("\nNaive Bayes predictions completed!")
#naive bayes accuracy
nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)
print("Naive Bayes Accuracy:")
print(nb_accuracy)

#naive bayes classification report
print("\n===================================================")
print("NAIVE BAYES CLASSIFICATION REPORT")
print("===================================================\n")
print(
    classification_report(
        y_test,
        nb_predictions
    )
)

#naive bayes confusion matrix
nb_cm = confusion_matrix(
    y_test,
    nb_predictions
)
plt.figure(figsize=(10, 7))
sns.heatmap(
    nb_cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.close()

#svm model
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
print("\nSVM model trained successfully!")
#svm model prediction
svm_predictions = svm_model.predict(X_test)
print("\nSVM predictions completed!")
#svm model accuracy
svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)
print("SVM Accuracy:")
print(svm_accuracy)

#svm model classification report
print("\n===================================================")
print("SVM CLASSIFICATION REPORT")
print("===================================================\n")
print(
    classification_report(
        y_test,
        svm_predictions
    )
)

#svm model confusion matrix
svm_cm = confusion_matrix(
    y_test,
    svm_predictions
)
plt.figure(figsize=(10, 7))
sns.heatmap(
    svm_cm,
    annot=True,
    fmt='d',
    cmap='Oranges'
)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.close()

#model comparison
models = ['Logistic Regression', 'Naive Bayes', 'SVM']
accuracies = [
    lr_accuracy,
    nb_accuracy,
    svm_accuracy
]
comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies
})
print(comparison_df)

#visualization
plt.figure(figsize=(8, 5))
sns.barplot(
    x='Model',
    y='Accuracy',
    data=comparison_df
)
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.close()

#best model selection
best_accuracy = max(accuracies)
best_model_name = models[
    accuracies.index(best_accuracy)
]
print("\nBest Performing Model:")
print(best_model_name)
print("\nBest Accuracy:")
print(best_accuracy)

# create path safely
model_path = os.path.join(model_dir, "emotion_model.pkl")
# save model
trained_models = {
    'Logistic Regression': lr_model,
    'Naive Bayes': nb_model,
    'SVM': svm_model
}
best_model = trained_models[best_model_name]
with open(model_path, "wb") as file:
    pickle.dump(best_model, file)
print("Emotion classification model saved successfully!")

#save label encoder
encoder_path = os.path.join(model_dir, "emotion_encoder.pkl")
with open(encoder_path, 'wb') as file:
    pickle.dump(emotion_encoder, file)
print("Emotion Label encoder saved successfully!")

#custom prediction
sample_text = "I feel very stressed and anxious"
# Preprocess text
sample_vector = vectorizer.transform([preprocess_sample_text(sample_text)])
# Predict
prediction = best_model.predict(sample_vector)
# Decode label
predicted_emotion = emotion_encoder.inverse_transform(
    prediction
)
print("\nPredicted Emotion:")
print(predicted_emotion[0])

#multiple sample prediction
sample_inputs = [
    "I feel very lonely",
    "I am extremely anxious today",
    "Nobody understands me",
    "I feel stressed because of studies",
    "I am feeling hopeless"
]
print("\n===================================================")
print("MULTIPLE SAMPLE PREDICTIONS")
print("===================================================\n")
for text in sample_inputs:
    vector = vectorizer.transform([preprocess_sample_text(text)])
    prediction = best_model.predict(vector)
    decoded_prediction = emotion_encoder.inverse_transform(
        prediction
    )
    print(f"Input: {text}")
    print(f"Predicted Emotion: {decoded_prediction[0]}")
    print("------------------------------------------------")

#final summary
print("\n===================================================")
print("PHASE 5 COMPLETED SUCCESSFULLY")
print("===================================================")

print(f"\nFinal Dataset Shape: {df.shape}")

print("\nTasks Completed:")
print("1. Dataset Loaded")
print("2. Labels Encoded")
print("3. TF-IDF Features Loaded")
print("4. Train-Test Split Completed")
print("5. Logistic Regression Trained")
print("6. Naive Bayes Trained")
print("7. SVM Trained")
print("8. Model Comparison Completed")
print("9. Confusion Matrix Created")
print("10. Best Model Saved")
print("11. Label Encoder Saved")
print("12. Custom Predictions Tested")
