import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import requests
from datetime import datetime
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="GRU Chatbot with Real-time Data",
    page_icon="💬",
    layout="wide"
)

class GRUChatbot:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.max_sequence_length = 100
        self.vocab_size = 10000
        self.model_vocab_size = None

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def create_model(self, vocab_size, embedding_dim=128, gru_units=256):
        # Single-step decoder predicting next token from last hidden state
        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=self.max_sequence_length),
            GRU(gru_units, return_sequences=False, dropout=0.3),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(vocab_size, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def prepare_training_data(self, conversations):
        input_texts = []
        target_texts = []
        for conversation in conversations:
            if len(conversation) >= 2:
                for i in range(len(conversation) - 1):
                    input_texts.append(self.preprocess_text(conversation[i]))
                    target_texts.append(self.preprocess_text(conversation[i + 1]))
        return input_texts, target_texts

    def train_model(self, input_texts, target_texts):
        # Tokenize
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        all_texts = input_texts + target_texts
        self.tokenizer.fit_on_texts(all_texts)

        # Convert to sequences
        input_sequences = self.tokenizer.texts_to_sequences(input_texts)
        target_sequences = self.tokenizer.texts_to_sequences(target_texts)

        # Guard: ensure we actually have tokens
        if len(input_sequences) == 0 or len(target_sequences) == 0:
            raise ValueError("No sequences produced from texts. Check your input data.")

        # Pad inputs
        X = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='post')

        # Pad targets
        y_full = pad_sequences(target_sequences, maxlen=self.max_sequence_length, padding='post')

        # CRITICAL FIX: use .size (int), not .shape (tuple)
        if getattr(y_full, "size", 0) > 0:
            # Next-token prediction: use first token of reply
            y = y_full[:, 0]
        else:
            y = np.zeros((0,), dtype=np.int64)

        # Model vocab size respecting tokenizer
        self.model_vocab_size = min(self.vocab_size, len(self.tokenizer.word_index) + 1)

        # Clip labels to valid range
        y = np.clip(y, 0, self.model_vocab_size - 1).astype(np.int64)

        # Remove PAD labels (0) to avoid learning PAD
        valid_mask = y != 0
        X = X[valid_mask]
        y = y[valid_mask]

        if X.size == 0 or y.size == 0:
            raise ValueError("No valid training samples after filtering PAD labels. Add more data or vary conversations.")

        # Train/validation split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build and train model
        self.model = self.create_model(self.model_vocab_size)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=32,
            verbose=0
        )
        return history

    def generate_response(self, input_text, max_length=50):
        if not self.model or not self.tokenizer:
            return "Model not trained yet!"

        processed_input = self.preprocess_text(input_text)
        sequence = self.tokenizer.texts_to_sequences([processed_input])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')

        response_words = []
        current_sequence = padded_sequence[0].tolist()
        index_word = self.tokenizer.index_word  # built-in reverse map

        for _ in range(max_length):
            preds = self.model.predict(np.array([current_sequence]), verbose=0)  # (1, vocab)
            next_word_idx = int(np.argmax(preds))

            if next_word_idx == 0 or next_word_idx not in index_word:
                break

            next_word = index_word[next_word_idx]
            if next_word == "<OOV>":
                break

            response_words.append(next_word)
            current_sequence = current_sequence[1:] + [next_word_idx]

        return " ".join(response_words) if response_words else "I'm not sure how to respond to that."

def get_real_time_data():
    data = {}
    try:
        # Weather requires a real API key; keep explicit message
        data['weather'] = "Weather data requires a valid OpenWeather API key."

        # Cryptocurrency data (Coindesk)
        crypto_url = "https://api.coindesk.com/v1/bpi/currentprice.json"
        crypto_response = requests.get(crypto_url, timeout=5)
        if crypto_response.status_code == 200:
            crypto_data = crypto_response.json()
            data['bitcoin_price'] = (
                crypto_data.get('bpi', {})
                           .get('USD', {})
                           .get('rate', 'Unavailable')
            )

        # Random quotes API
        quote_url = "https://api.quotable.io/random"
        quote_response = requests.get(quote_url, timeout=5)
        if quote_response.status_code == 200:
            quote_data = quote_response.json()
            content = quote_data.get('content')
            author = quote_data.get('author')
            if content and author:
                data['quote'] = f"{content} - {author}"

    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")

    return data

def create_sample_conversations():
    conversations = [
        ["Hello", "Hi there! How can I help you today?"],
        ["How are you?", "I'm doing well, thank you for asking!"],
        ["What's the weather like?", "I can help you with weather information if you provide a location."],
        ["Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"],
        ["What time is it?", f"The current time is {datetime.now().strftime('%H:%M:%S')}"],
        ["Goodbye", "Goodbye! Have a great day!"],
        ["What can you do?", "I can chat with you, provide information, and learn from our conversations!"],
        ["Thank you", "You're welcome! I'm happy to help."],
        ["What's Bitcoin price?", "Let me check the current Bitcoin price for you."],
        ["Tell me a quote", "Here's an inspiring quote for you."]
    ]
    extended_conversations = [
        ["Good morning", "Good morning! Ready to start the day?"],
        ["How's the market?", "I can provide current market information if you specify what you're interested in."],
        ["I'm feeling sad", "I'm sorry to hear that. Would you like to talk about what's bothering you?"],
        ["What's new today?", "Let me check the latest updates for you."],
        ["Can you help me?", "Of course! I'm here to help. What do you need assistance with?"]
    ]
    return conversations + extended_conversations

def main():
    st.title("GRU Chatbot with Real-time Data")
    st.markdown("---")

    # Initialize state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = GRUChatbot()
        st.session_state.trained = False
        st.session_state.chat_history = []
        st.session_state.real_time_data = {}

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.caption("Train model, then chat. Optional: add your OpenWeather API key in code.")

    # Model training
    st.subheader("Model Training")
    if st.button("Train Model"):
        with st.spinner("Training GRU model..."):
            conversations = create_sample_conversations()
            input_texts, target_texts = st.session_state.chatbot.prepare_training_data(conversations)
            try:
                _ = st.session_state.chatbot.train_model(input_texts, target_texts)
                st.session_state.trained = True
                st.success("Model trained successfully!")
            except Exception as e:
                st.session_state.trained = False
                st.error(f"Training failed: {e}")

    # Real-time data
    st.subheader("Real-time Data")
    if st.button("Fetch Real-time Data"):
        with st.spinner("Fetching data..."):
            real_time_data = get_real_time_data()
            st.session_state.real_time_data = real_time_data
            for key, value in real_time_data.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")

    # Clear chat
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat Interface")

        if st.session_state.trained:
            st.success("Model is trained and ready!")
        else:
            st.warning("Please train the model first.")

        # Chat history
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                st.write(f"**You:** {user_msg}")
                st.write(f"**Bot:** {bot_msg}")
                st.markdown("---")

        user_input = st.text_input("Type your message:", key="user_input")
        if st.button("Send") and user_input:
            if st.session_state.trained:
                bot_response = st.session_state.chatbot.generate_response(user_input)

                lowered = user_input.lower()
                if any(keyword in lowered for keyword in ['bitcoin', 'btc', 'price', 'quote', 'weather']):
                    real_time_data = get_real_time_data()
                    if any(k in lowered for k in ['bitcoin', 'btc', 'price']):
                        if 'bitcoin_price' in real_time_data:
                            bot_response = f"Current Bitcoin price: {real_time_data['bitcoin_price']}"
                    elif 'quote' in lowered:
                        if 'quote' in real_time_data:
                            bot_response = real_time_data['quote']
                    elif 'weather' in lowered:
                        bot_response = real_time_data.get(
                            'weather',
                            "Weather service unavailable right now."
                        )

                st.session_state.chat_history.append((user_input, bot_response))
                st.rerun()
            else:
                st.error("Please train the model first!")

    with col2:
        st.subheader("Analytics")

        if st.session_state.chat_history:
            total_messages = len(st.session_state.chat_history)
            st.metric("Total Conversations", total_messages)

            user_msg_lengths = [len(pair[0]) for pair in st.session_state.chat_history]
            bot_msg_lengths = [len(pair) for pair in st.session_state.chat_history]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=user_msg_lengths,
                mode='lines+markers',
                name='User Message Length',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=bot_msg_lengths,
                mode='lines+markers',
                name='Bot Message Length',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="Message Length Trends",
                xaxis_title="Message Number",
                yaxis_title="Character Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Information")
        st.write("Architecture: GRU-based Neural Network")
        st.write("Max Sequence Length: 100")
        st.write(f"Vocabulary Size (model): {st.session_state.chatbot.model_vocab_size or 10000}")
        st.write("Embedding Dimension: 128")
        st.write("GRU Units: 256")

    st.markdown("---")
    st.markdown(
        "Note: This chatbot uses GRU neural networks and can integrate real-time data. "
        "The model learns from conversation patterns and can be enhanced with larger datasets."
    )

if __name__ == "__main__":
    main()
