print ('Phase 3 --> NLP Precessing \n')
#import libraries
import os
import pandas as pd
import numpy as np
import re
import string
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
print("Libraries imported successfully!")
# Display all columns
pd.set_option('display.max_columns', None)
#download nltk resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("NLTK resources downloaded successfully!")

#load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
df = pd.read_csv(os.path.join(data_dir, "eda_processed_dataset.csv"))
print("Dataset loaded successfully!")

#initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
print("NLP tools initialized successfully!")

#NLP(natural language processing)

def preprocess_text(text):

# convert to string
    text = str(text)

#lowercase
    text = text.lower()

#remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

#remove numbers
    text = re.sub(r'\d+', '', text)

#remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

#remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

#remove extra spaces
    text = text.strip()

#tokenization
    tokens = word_tokenize(text)

#stopwords removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]
#lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]
#join words
    cleaned_text = " ".join(tokens)
    return cleaned_text

#apply processing
df['processed_text'] = df['text'].apply(
    preprocess_text
)
print("Text preprocessing completed successfully!")

#original vs processed text
for i in range(5):
    print("\n================================================")
    print("ORIGINAL TEXT:\n")
    print(df['text'].iloc[i])
    print("\nPROCESSED TEXT:\n")
    print(df['processed_text'].iloc[i])

#empty text check
empty_rows = df[
    df['processed_text'].str.len() == 0
]
print("Number of Empty Processed Rows:")
print(len(empty_rows))

#remove empty rows
df = df[
    df['processed_text'].str.len() > 0
]

#processed text length
print('Processed Text Length:')
df['processed_length'] = df['processed_text'].apply(len)
print(df['processed_length'].describe())

#visualization
plt.figure(figsize=(10, 5))
plt.hist(df['processed_length'], bins=30)
plt.title("Processed Text Length Distribution")
plt.xlabel("Processed Length")
plt.ylabel("Frequency")
result_dir = os.path.join(script_dir, "..", "results")
plt.savefig(
    os.path.join(result_dir, "processed_text_length_distribution.png")
)
plt.close()

#TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=5000
)
X = vectorizer.fit_transform(
    df['processed_text']
)
print("TF-IDF Vectorization Completed!")
print("\nShape of TF-IDF Matrix:")
print(X.shape)

#feature names
feature_names = vectorizer.get_feature_names_out()
print("\nFirst 20 Features:\n")
print(feature_names[:20])

#convert TF-IDF to data frame
tfidf_df = pd.DataFrame(
    X.toarray(),
    columns=feature_names
)
print("\nTF-IDF DataFrame Created Successfully!")
print("\nTF-IDF DataFrame Shape:")
print(tfidf_df.shape)

#disply TF-IDF sample
print("\nTF-IDF Sample:\n")
print(tfidf_df.head())

#top TF-IDF features
feature_sums = np.array(X.sum(axis=0)).flatten()
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Score': feature_sums
})
top_features = feature_importance.sort_values(
    by='Score',
    ascending=False
).head(20)
print("\nTop 20 Important TF-IDF Features:\n")
print(top_features)

#top TF-IDF visualization
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Score',
    y='Feature',
    data=top_features
)
plt.title("Top 20 TF-IDF Features")
plt.xlabel("TF-IDF Score")
plt.ylabel("Features")
plt.savefig(
    os.path.join(result_dir, "top_tfidf_features.png")
)
plt.close()

#save TF-IDF vectorizer
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(
    script_dir,
    "..",
    "models",
    "tfidf_vectorizer.pkl"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'wb') as file:
    pickle.dump(vectorizer, file)
print("\nTF-IDF Vectorizer Saved Successfully!")

#save TD-IDF matrix
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(
    script_dir,
    "..",
    "models",
    "tfidf_features.pkl"
)
with open(output_path, 'wb') as file:
    pickle.dump(X, file)
print("\nTF-IDF Matrix Saved Successfully!")

#save processed dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(
    script_dir,
    "..",
    "data",
    "final_processed_dataset.csv"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
try:
    df.to_csv(output_path, index=False, encoding="utf-8")
except PermissionError:
    print("\nCould not save processed dataset.")
    print("Please close final_processed_dataset.csv if it is open in Excel, VS Code, or another app.")
    print(f"File path: {os.path.abspath(output_path)}")
    raise
print("Processed dataset saved successfully!")

# ============================================================
# STEP 30 — FINAL SUMMARY
# ============================================================

print("\n===================================================")
print("PHASE 3 COMPLETED SUCCESSFULLY")
print("===================================================")

print(f"\nFinal Dataset Shape: {df.shape}")

print("\nTasks Completed:")
print("1. Dataset Loaded")
print("2. NLP Preprocessing Pipeline Created")
print("3. Lowercasing Applied")
print("4. Punctuation Removed")
print("5. URLs Removed")
print("6. Numbers Removed")
print("7. Stopwords Removed")
print("8. Lemmatization Applied")
print("9. TF-IDF Vectorization Completed")
print("10. Vectorizer Saved")
print("11. Processed Dataset Saved")
