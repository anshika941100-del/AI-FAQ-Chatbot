import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# FAQ Dataset
faqs = [
    {"question": "What is Artificial Intelligence?", "answer": "Artificial Intelligence is the simulation of human intelligence by machines."},
    {"question": "What is Machine Learning?", "answer": "Machine Learning is a subset of AI that enables systems to learn from data."},
    {"question": "What is Deep Learning?", "answer": "Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers."},
    {"question": "What is Natural Language Processing?", "answer": "Natural Language Processing allows computers to understand and process human language."},
    {"question": "What is Computer Vision?", "answer": "Computer Vision enables computers to interpret and analyze images and videos."},
    {"question": "What is a Neural Network?", "answer": "A Neural Network is a computing system inspired by the structure of the human brain."},
    {"question": "What is supervised learning?", "answer": "Supervised learning uses labeled data to train machine learning models."},
    {"question": "What is unsupervised learning?", "answer": "Unsupervised learning finds hidden patterns in unlabeled data."},
    {"question": "What is reinforcement learning?", "answer": "Reinforcement learning trains agents using rewards and penalties."},
    {"question": "What is data preprocessing?", "answer": "Data preprocessing involves cleaning and transforming raw data before analysis."},
    {"question": "What is overfitting?", "answer": "Overfitting occurs when a model performs well on training data but poorly on unseen data."},
    {"question": "What is underfitting?", "answer": "Underfitting occurs when a model fails to capture important patterns in data."},
    {"question": "What is TensorFlow?", "answer": "TensorFlow is an open-source machine learning framework developed by Google."},
    {"question": "What is Python used for in AI?", "answer": "Python is widely used in AI due to its simplicity and powerful libraries."},
    {"question": "What are chatbots?", "answer": "Chatbots are software applications that interact with users through text or voice conversations."},
    {"question": "What is a dataset?", "answer": "A dataset is a collection of data used for analysis, training, and testing models."},
    {"question": "What is feature engineering?", "answer": "Feature engineering is the process of selecting and transforming variables for machine learning models."},
    {"question": "What is model accuracy?", "answer": "Accuracy measures the percentage of correct predictions made by a model."},
    {"question": "What is a training set?", "answer": "A training set is the portion of data used to train a machine learning model."},
    {"question": "What is a test set?", "answer": "A test set is the portion of data used to evaluate a machine learning model's performance."}
]

# NLP Preprocessing Function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    return " ".join(tokens)

# Preprocess FAQ questions
questions = [preprocess(faq["question"]) for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Streamlit UI
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")

st.title("🤖 FAQ Chatbot")
st.write("Ask any question related to Artificial Intelligence.")

user_query = st.text_input("Enter your question:")

if user_query:
    processed_query = preprocess(user_query)

    user_vector = vectorizer.transform([processed_query])

    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match = similarity_scores.argmax()

    st.subheader("Answer")
    st.success(answers[best_match])

st.markdown("---")
st.subheader("Available FAQ Questions")

for faq in faqs:
    st.write("•", faq["question"])