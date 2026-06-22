# IMPORT LIBRARIES
import os
import pickle
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
print("Libraries imported successfully.")

# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# data and models folders
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(os.path.join(DATA_DIR, "final_processed_dataset.csv"))
print("Dataset loaded successfully.")

# =====================================================
# LOAD VECTORIZER
# =====================================================

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as file:
    vectorizer = pickle.load(file)
print("Vectorizer loaded successfully.")

# =====================================================
# LOAD INTENT MODEL
# =====================================================

with open(os.path.join(MODEL_DIR, "intent_model.pkl"), "rb") as file:
    intent_model = pickle.load(file)
print("Intent model loaded successfully.")

# =====================================================
# LOAD INTENT ENCODER
# =====================================================

with open(os.path.join(MODEL_DIR, "intent_encoder.pkl"), "rb") as file:
    intent_encoder = pickle.load(file)
print("Intent encoder loaded successfully.")

# =====================================================
# LOAD EMOTION MODEL
# =====================================================

with open(os.path.join(MODEL_DIR, "emotion_model.pkl"), "rb") as file:
    emotion_model = pickle.load(file)
print("Emotion model loaded successfully.")

# =====================================================
# LOAD EMOTION ENCODER
# =====================================================

with open(os.path.join(MODEL_DIR, "emotion_encoder.pkl"), "rb") as file:
    emotion_encoder = pickle.load(file)
print("Emotion encoder loaded successfully.")

# PREPARE DATA
X = df["processed_text"]

# INTENT DATA
y_intent = intent_encoder.transform(
    df["intent"]
)

# EMOTION DATA

y_emotion = emotion_encoder.transform(
    df["emotion"]
)

# TRAIN TEST SPLIT
X_train,X_test,y_intent_train,y_intent_test = train_test_split(
    X,
    y_intent,
    test_size=0.2,
    random_state=42
)

# TF-IDF TRANSFORM

X_test_tfidf = vectorizer.transform(
    X_test
)

# INTENT PREDICTION
intent_predictions = intent_model.predict(
    X_test_tfidf
)

# INTENT ACCURACY

intent_accuracy = accuracy_score(
    y_intent_test,
    intent_predictions
)
print("\n==============================")
print("INTENT MODEL RESULTS")
print("==============================")
print(
    f"Accuracy: {intent_accuracy:.4f}"
)
#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
print("Total intent classes:",
      len(intent_encoder.classes_))

print("Classes in test set:",
      len(set(y_intent_test)))

print("Intent class names:")
print(intent_encoder.classes_)

# INTENT REPORT
intent_report = classification_report(
    y_intent_test,
    intent_predictions,
    labels=range(len(intent_encoder.classes_)),
    target_names=intent_encoder.classes_,
    zero_division=0
)
print(intent_report)

# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# metrics folder path
METRICS_DIR = os.path.join(PROJECT_DIR, "metrics")

# create folder if it doesn't exist
os.makedirs(METRICS_DIR, exist_ok=True)

# save intent report
report_path = os.path.join(METRICS_DIR, "intent_report.txt")

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        f"Accuracy: {intent_accuracy:.4f}\n\n"
    )
    file.write(intent_report)

print("Intent report saved successfully!")

# INTENT CONFUSION MATRIX
intent_cm = confusion_matrix(
    y_intent_test,
    intent_predictions
)

plt.figure(figsize=(12,8))
sns.heatmap(
    intent_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title(
    "Intent Confusion Matrix"
)
plt.xlabel(
    "Predicted"
)
plt.ylabel(
    "Actual"
)
plt.tight_layout()
# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# results folder path
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# create folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# save plot
plt.savefig(
    os.path.join(RESULTS_DIR, "intent_confusion_matrix.png")
)

print("Confusion matrix saved successfully!")
plt.close()

# EMOTION SPLIT
X_train,X_test,y_emotion_train,y_emotion_test = train_test_split(
    X,
    y_emotion,
    test_size=0.2,
    random_state=42
)

# TF-IDF TRANSFORM
X_test_tfidf = vectorizer.transform(
    X_test
)

# EMOTION PREDICTIONS
emotion_predictions = emotion_model.predict(
    X_test_tfidf
)

# EMOTION ACCURACY
emotion_accuracy = accuracy_score(
    y_emotion_test,
    emotion_predictions
)
print("\n==============================")
print("EMOTION MODEL RESULTS")
print("==============================")
print(
    f"Accuracy: {emotion_accuracy:.4f}"
)

# EMOTION REPORT
emotion_report = classification_report(
    y_emotion_test,
    emotion_predictions,
    labels=range(len(emotion_encoder.classes_)),
    target_names=emotion_encoder.classes_,
    zero_division=0
)
print(emotion_report)

# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# metrics folder path
METRICS_DIR = os.path.join(PROJECT_DIR, "metrics")

# create folder if it doesn't exist
os.makedirs(METRICS_DIR, exist_ok=True)

# save emotion report
report_path = os.path.join(METRICS_DIR, "emotion_report.txt")

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:
    file.write(
        f"Accuracy: {emotion_accuracy:.4f}\n\n"
    )
    file.write(emotion_report)

print("Emotion report saved successfully!")
# EMOTION CONFUSION MATRIX
emotion_cm = confusion_matrix(
    y_emotion_test,
    emotion_predictions
)
plt.figure(figsize=(12,8))
sns.heatmap(
    emotion_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)
plt.title(
    "Emotion Confusion Matrix"
)
plt.xlabel(
    "Predicted"
)
plt.ylabel(
    "Actual"
)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "emotional_confusion_matrix.png")
)

plt.close()

def preprocess_user_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return " ".join(text.split())

# HUMAN EVALUATION TEMPLATE
evaluation_data = {
    "User_Message": [
        "I feel stressed about exams",
        "Nobody understands me",
        "I feel anxious today",
        "I feel lonely",
        "I am happy today"
    ],
    "Expected_Intent": [
        "seeking_help",
        "low_self_esteem",
        "seeking_help",
        "seeking_help",
        "positive"
    ],
    "Expected_Emotion": [
        "stress",
        "sadness",
        "anxiety",
        "loneliness",
        "happy"
    ],
    "Predicted_Intent": [],
    "Predicted_Emotion": [],
    "Bot_Response": [],
    "Relevance_Score": [],
    "Supportive_Score": []
}

supportive_responses = {
    "stress": "Exams and pressure can feel heavy. Try one small task first, then take a short break.",
    "anxiety": "Take a slow breath and focus on one thing you can control right now.",
    "loneliness": "Feeling alone can hurt. Reaching out to one safe person may help.",
    "sadness": "I'm sorry you're feeling this way. Be gentle with yourself and consider telling someone you trust.",
    "sad": "I'm sorry you're feeling this way. Be gentle with yourself and consider telling someone you trust.",
    "happy": "That's good to hear. Notice what helped you feel this way today.",
    "neutral": "I hear you. Try taking one small step that feels manageable right now."
}

for index, message in enumerate(evaluation_data["User_Message"]):
    processed_message = preprocess_user_text(message)
    message_vector = vectorizer.transform([processed_message])

    intent_prediction = intent_model.predict(message_vector)
    predicted_intent = intent_encoder.inverse_transform(intent_prediction)[0]

    emotion_prediction = emotion_model.predict(message_vector)
    predicted_emotion = emotion_encoder.inverse_transform(emotion_prediction)[0]

    bot_response = supportive_responses.get(
        str(predicted_emotion).lower(),
        "I hear you. Try taking one small step that feels manageable right now."
    )

    expected_intent = evaluation_data["Expected_Intent"][index]
    expected_emotion = evaluation_data["Expected_Emotion"][index]
    relevance_score = 5
    if predicted_intent != expected_intent:
        relevance_score -= 1
    if predicted_emotion != expected_emotion:
        relevance_score -= 1

    supportive_words = [
        "sorry",
        "hear",
        "help",
        "try",
        "safe",
        "gentle",
        "breath",
        "step"
    ]
    supportive_score = 3 + min(
        2,
        sum(word in bot_response.lower() for word in supportive_words)
    )

    evaluation_data["Predicted_Intent"].append(predicted_intent)
    evaluation_data["Predicted_Emotion"].append(predicted_emotion)
    evaluation_data["Bot_Response"].append(bot_response)
    evaluation_data["Relevance_Score"].append(relevance_score)
    evaluation_data["Supportive_Score"].append(supportive_score)

evaluation_df = pd.DataFrame(
    evaluation_data
)
# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)

# results folder path
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# create folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# save CSV file
template_path = os.path.join(RESULTS_DIR, "human_evaluation_template.csv")
report_path = os.path.join(RESULTS_DIR, "human_evaluation_report.csv")

try:
    evaluation_df.to_csv(
        template_path,
        index=False,
        encoding="utf-8"
    )
    evaluation_df.to_csv(
        report_path,
        index=False,
        encoding="utf-8"
    )
except PermissionError:
    print("\nCould not update the human evaluation files.")
    print("Please close them if they are open in Excel, VS Code, or another app.")
    print(f"Template path: {template_path}")
    print(f"Report path: {report_path}")
    raise

print("Human evaluation template saved successfully!")
print(f"Template path: {template_path}")
print("Human evaluation report saved successfully!")
print(f"Report path: {report_path}")

# FINISHED
print("\nEvaluation Completed Successfully!")
