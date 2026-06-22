# =====================================================
# PHASE 8: GRADIO UI FOR AI MENTAL HEALTH CHATBOT
# =====================================================

import gradio as gr
import os
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

# Load DialogPT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)

# =====================================================
# SUPPORT TEMPLATES
# =====================================================

def preprocess_user_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    #stopword removal and lemmatization
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_tokens = [word for word in word_tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    text = " ".join(lemmatized_tokens)

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return " ".join(text.split())

def emotion_keyword_override(text, predicted_emotion):
    text = str(text).lower()
    if any(word in text for word in ["mad", "angry", "anger", "aggressive", "agressively", "beat someone"]):
        return "anger"
    if any(word in text for word in ["lonely", "loneliness", "alone", "isolated"]):
        return "loneliness"
    if any(word in text for word in ["anxious", "anxiety", "panic", "worried", "nervous"]):
        return "anxiety"
    if any(word in text for word in ["cry", "crying", "sad", "upset", "hopeless", "bad"]):
        return "sadness"
    if re.search(r'\b(no|not|without)\s+(stress|stressed|exam|exams|overwhelmed)\b', text):
        return predicted_emotion
    if any(word in text for word in ["stress", "stressed", "exam", "exams", "overwhelmed"]):
        return "stress"
    if re.search(r'\b(they|he|she|someone|others)\s+(are|is|seem|seems|look|looks)\s+happy\b', text):
        return predicted_emotion
    if re.search(r'\b(happy|glad|great|better|good)\b', text):
        if re.search(r'\b(not me|not happy|no happy)\b', text):
            return predicted_emotion
        return "happy"
    return predicted_emotion

def generate_dialogpt_response(text, history=""):
    """Generate a response using DialogPT medium model."""
    try:
        # Combine history and current input
        combined_text = history + text if history else text
        
        # Encode the input text with EOS token
        input_ids = dialogpt_tokenizer.encode(combined_text + dialogpt_tokenizer.eos_token, return_tensors="pt").to(device)
        
        # Generate response
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
        
        chat_history_ids = dialogpt_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=150,
            pad_token_id=dialogpt_tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
        
        # Decode the response
        response = dialogpt_tokenizer.decode(
            chat_history_ids[:, input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        
        return response.strip() if response.strip() else None
    except Exception as e:
        print(f"DialogPT generation error: {e}")
        return None

def get_intent_guidance(intent):
    intent_text = str(intent).lower()
    if "advice" in intent_text or "help" in intent_text:
        return "Offer gentle, practical next steps."
    if "vent" in intent_text or "share" in intent_text:
        return "Validate the user's feelings and invite them to share more."
    if "question" in intent_text or "information" in intent_text:
        return "Answer clearly while staying supportive."
    if "crisis" in intent_text or "suicide" in intent_text or "self" in intent_text:
        return "Encourage immediate support from trusted people or professionals."
    return "Match the user's intent and keep the response supportive."

# =====================================================
# CRISIS DETECTION
# =====================================================

def detect_crisis(text):
    """Enhanced crisis detection with more keywords."""
    text = text.lower()
    crisis_keywords = [
        "suicide", "suicidal", "kill myself",
        "self harm", "self-harm", "hurt myself",
        "end my life", "end life", "don't want to live",
        "want to die", "should be dead", "better off dead",
        "hang myself", "overdose", "cut myself", "slice wrist",
        "jump", "goodbye world", "farewell", "last message"
    ]
    return any(keyword in text for keyword in crisis_keywords)

# =====================================================
# MAIN CHAT FUNCTION
# =====================================================

def chatbot_response(user_input, history):

    # CRISIS CHECK (with enhanced resources)
    if detect_crisis(user_input):
        return (
            "⚠ **I am really concerned about you.**\n\n"
            "Please reach out to a trusted person or professional immediately.\n\n"
            "**Crisis Resources:**\n"
            "🆘 National Suicide Prevention Lifeline: **1-800-273-8255**\n"
            "📱 Crisis Text Line: Text **'HELLO'** to **741741**\n"
            "🌍 International: https://www.iasp.info/resources/Crisis_Centres/"
        )

    # INTENT PREDICTION
    processed_input = preprocess_user_text(user_input)
    intent_vec = vectorizer.transform([processed_input])
    intent_pred = intent_model.predict(intent_vec)
    intent = intent_encoder.inverse_transform(intent_pred)[0]

    # EMOTION PREDICTION
    emotion_pred = emotion_model.predict(intent_vec)
    emotion = emotion_encoder.inverse_transform(emotion_pred)[0]
    emotion = emotion_keyword_override(user_input, emotion)

    # DIALOGPT RESPONSE (with minimal fallback)
    history_text = ""
    try:
        if history:
            # Get last 3 turns safely
            for h in history[-3:]:
                try:
                    if isinstance(h, (tuple, list)) and len(h) >= 2:
                        prev_user = str(h[0]) if h[0] else ""
                        prev_bot = str(h[1]) if h[1] else ""
                        if prev_user and prev_bot:
                            history_text += f"User: {prev_user} Bot: {prev_bot[:80]} "
                except Exception as e:
                    print(f"History parsing warning: {e}")
                    continue
    except Exception as e:
        print(f"History error: {e}")
        history_text = ""
    
    try:
        dialogpt_response = generate_dialogpt_response(user_input, history_text)
    except Exception as e:
        print(f"DialogPT error: {e}")
        dialogpt_response = None
    
    # Minimal fallback (only if DialogPT completely fails)
    if not dialogpt_response or dialogpt_response.strip() == "":
        dialogpt_response = "I appreciate you sharing that with me."

    response = f"""**Intent:** {intent}  
**Emotion:** {emotion}

**Bot:** {dialogpt_response}"""

    return response

# =====================================================
# GRADIO UI
# =====================================================

with gr.Blocks() as app:

    gr.Markdown("""
    # 🧠 AI Mental Health Support Chatbot

    ⚠ **Disclaimer:**  
    This chatbot provides emotional support only.  
    It is NOT a licensed therapist or medical professional.
    """)

    chatbot = gr.ChatInterface(
        fn=chatbot_response
    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    app.launch()
