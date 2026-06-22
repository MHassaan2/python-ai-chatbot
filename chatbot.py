print('Phase 6 --> Transformer Chatbot')
#import libraries
import pickle
import os
import torch
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import warnings
warnings.filterwarnings('ignore')
print('Libraries imported successfully!')
# =====================================================
# LOAD MODELS
# =====================================================
# get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# go to project root
PROJECT_DIR = os.path.dirname(BASE_DIR)
# models folder path
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# load intent model
intent_model = pickle.load(
    open(os.path.join(MODEL_DIR, "intent_model.pkl"), "rb")
)
# load emotion model
emotion_model = pickle.load(
    open(os.path.join(MODEL_DIR, "emotion_model.pkl"), "rb")
)
# load TF-IDF vectorizer
vectorizer = pickle.load(
    open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb")
)
# load intent encoder
intent_encoder = pickle.load(
    open(os.path.join(MODEL_DIR, "intent_encoder.pkl"), "rb")
)
# load emotion encoder
emotion_encoder = pickle.load(
    open(os.path.join(MODEL_DIR, "emotion_encoder.pkl"), "rb")
)
#conversation memory
chat_history_ids = None
#load microsoft DialoGPT model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
print('DialoGPT model and tokenizer loaded successfully!')
#NLP
def preprocess_user_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]
    return ' '.join(tokens)
#predict intent function
def predict_intent(text):
    vector = vectorizer.transform([text])
    prediction = intent_model.predict(vector)
    intent = intent_encoder.inverse_transform(
        prediction
    )[0]
    return intent
#predict emotion function
def predict_emotion(text):
    vector = vectorizer.transform([text])
    prediction = emotion_model.predict(vector)
    emotion = emotion_encoder.inverse_transform(
        prediction
    )[0]
    return emotion
#generate dialogue response function
def generate_dialogpt_response(text,emotion):
    global chat_history_ids
    prompt = f"""
User emotion: {emotion}
User says:
{text}
Respond supportively:
"""
    new_input_ids = tokenizer.encode(
        prompt + tokenizer.eos_token,
        return_tensors="pt"
    )
    bot_input_ids = (
        torch.cat(
            [chat_history_ids, new_input_ids],
            dim=-1
        )
        if chat_history_ids is not None
        else new_input_ids
    )
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=700,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=40,
        top_p=0.9,
        temperature=0.6
    )
    response = tokenizer.decode(
        chat_history_ids[
            :,
            bot_input_ids.shape[-1]:
        ][0],
        skip_special_tokens=True
    )
    return response
#create supportive response
emotion_templates = {
    "sadness":
    "I'm sorry you're feeling sad. ",
    "anxiety":
    "It's understandable to feel anxious. ",
    "stress":
    "Stress can feel overwhelming sometimes. ",
    "loneliness":
    "Feeling lonely can be difficult. ",
    "depression":
    "I'm here to listen and support you. ",
    "neutral":
    ""
}
#main chatbot function
def chatbot_response(user_text):
    intent = predict_intent(user_text)
    emotion = predict_emotion(user_text)
    generated_response = generate_dialogpt_response(user_text,emotion)
    supportive_prefix = emotion_templates.get(emotion,"")
    final_response = (
        supportive_prefix
        +
        generated_response
    )
    return {
        "intent": intent,
        "emotion": emotion,
        "response": final_response
    }
#test chatbot with sample input
user_input = "I feel stressed because of exams"
result = chatbot_response(user_input)
print("Intent:")
print(result["intent"])
print("\nEmotion:")
print(result["emotion"])
print("\nResponse:")
print(result["response"])
#interactive chat loop
print("""
==================================================

AI Mental Health Support Chatbot

Type 'exit' to quit.

This chatbot provides emotional support only.
It is not a licensed therapist.

==================================================
""")

while True:
    user_input = input('\n'"You: ")
    print(user_input)
    if user_input.lower() == "exit":
        print("\nChat ended.")
        break
    result = chatbot_response(
        user_input
    )
    print("\nIntent :", result["intent"])
    print("Emotion:", result["emotion"])
    print("\nChatbot:")
    print(result["response"])