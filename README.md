# Python AI Chatbot 🤖🗣️

This repository hosts an advanced Python-based AI Chatbot, engineered to engage in intelligent conversations by discerning user intent and emotional tone. It represents a comprehensive solution, spanning the entire machine learning lifecycle from data acquisition and preprocessing to model training, evaluation, and deployment of an interactive conversational agent. Leveraging cutting-edge Natural Language Processing (NLP) techniques and robust machine learning models, this chatbot aims to provide contextually aware, empathetic, and highly responsive interactions.

## Project Overview ✨🧠

The `python-ai-chatbot` project focuses on developing a sophisticated conversational AI capable of understanding and responding dynamically. Key features and components include:

*   **Data Preprocessing**: Robust pipelines to clean, transform, and prepare raw textual data for model training. 🧹📊
*   **Exploratory Data Analysis (EDA)**: Scripts to gain insights into the dataset, understand linguistic patterns, and inform feature engineering. 🔎📈
*   **Intent Recognition**: Machine learning models trained to classify the underlying purpose or goal behind a user's utterance. 🎯
*   **Emotion Detection**: Algorithms designed to identify and categorize the emotional state expressed in user input, enabling more empathetic responses. ❤️😠
*   **Model Training & Evaluation**: Dedicated modules for training intent and emotion classification models, along with comprehensive evaluation metrics to assess performance. 🧠✅
*   **Interactive Interface**: A user-friendly web interface built with Gradio, allowing seamless interaction with the chatbot. 💬💻
*   **Modular Design**: A well-structured codebase facilitating ease of understanding, maintenance, and future enhancements. 🏗️
## Screenshot 📸

<p align="center">
  <img src="Screenshot 2026-06-06 021809.png" width="800">
</p>

## Installation ⚙️🚀

To set up the project locally and get the AI chatbot running, please follow these steps:

1.  **Clone the Repository** 📁
    ```bash
    git clone https://github.com/MHassaan2/python-ai-chatbot.git
    cd python-ai-chatbot
    ```

2.  **Create and Activate a Virtual Environment** 🐍
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install Required Dependencies** 📦
    The project relies on several Python libraries. If a `requirements.txt` file is not present, please create one with the following content and then install:
    ```
    pandas
    numpy
    scikit-learn
    nltk
    gradio
    ```
    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data** 📚
    If NLTK is used for text processing (e.g., tokenization, stop words), download the necessary datasets:
    ```bash
    python -m nltk.downloader stopwords punkt wordnet
    ```
    *You may need to run this command from within a Python interpreter or adjust the script to automatically download these if they are not found.*

5.  **Run Data Preprocessing and Model Training (Optional)** 🧠
    If you wish to retrain the models or process the dataset from scratch, execute the following scripts in order:
    *   `python processing.py` (For initial data cleaning and feature engineering)
    *   `python train.py` (To train the intent and emotion classification models)
    *   `python evaluation.py` (To evaluate the trained models and generate reports)
    
6. **File Structure** 📦
   Your file structure should be:
   
```   
 AI_Mental_Health_Chatbot/
│
├── data/
│   ├── dataset.csv
|   ├── cleaned_dataset.csv
|   ├── eda_processed_dataset.csv
│   └── final_processed_dataset.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
|   ├── tfidf_features.pkl
│   ├── intent_model.pkl
│   ├── intent_encoder.pkl
│   ├── emotion_model.pkl
│   └── emotion_encoder.pkl
│
├── src/
│   ├── processing.py
│   ├── data understanding & EDA.py
│   ├── train.py
│   ├── chatbot.py
│   └── evaluation.py
│
├── app/
│   └── gradio_app.py
│
├── results/
│  
│   
└── metrics/
    ├── intent_report.txt
    └── emotion_report.txt
```

   *Note: Pre-trained model files (.pkl) are included in the repository. If you only want to run the chatbot, simply ensure the required model files are available and run app/gradio_app.py. Training and preprocessing scripts are only needed if you want to retrain the models or reproduce the complete workflow.*

## Usage 💬💻

Once installed, you can launch and interact with the AI chatbot through its Gradio interface.

1.  **Run the Chatbot Application** 🚀
    Execute the Gradio application script from your terminal:
    ```bash
    python gradio_app.py
    ```

2.  **Access the User Interface** 🌐
    Upon running the script, Gradio will provide a local URL (typically `http://127.0.0.1:7860`). Open this URL in your web browser to access the chatbot's interface.

3.  **Interact with the Chatbot** 🗣️
    Type your messages into the input field provided on the web page and press Enter or click the "Send" button. The chatbot will process your input, determine your intent and emotion, and generate a relevant response.

## Contributing 🤝💡

We welcome contributions to enhance the Python AI Chatbot! If you have ideas for improvements, bug fixes, or new features, please follow these steps:

1.  **Fork the Repository** 🍴
    Click the 'Fork' button at the top right of this repository's GitHub page.

2.  **Create a New Branch** 🌿
    Clone your forked repository, then create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    # Or for a bug fix:
    git checkout -b bugfix/issue-description
    ```

3.  **Make Your Changes** ✨
    Implement your improvements or fixes. Ensure your code adheres to the existing style and conventions.

4.  **Commit Your Changes** 💾
    Write clear and concise commit messages:
    ```bash
    git commit -m "feat: Add new awesome feature"
    # Or for a bug fix:
    git commit -m "fix: Resolve issue with [specific problem]"
    ```

5.  **Push to Your Branch** 📤
    ```bash
    git push origin feature/your-feature-name
    ```

6.  **Open a Pull Request** ✅
    Navigate to your forked repository on GitHub and open a pull request to the `main` branch of the original repository. Provide a detailed description of your changes.

Thank you for contributing!
