import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the tokenizer
#with open('tokenizer.json', 'r') as json_file:
    #tokenizer = tokenizer_from_json(json_file.read())

# Load the trained model
#model = load_model('sentiment_analysis_model.h5')

# Function to preprocess the input text
def preprocess_text(text):
    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    # Pad the sequence
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
    return sequence

# Function to predict the sentiment
def predict_sentiment(text):
    # Preprocess the text
    sequence = preprocess_text(text)
    # Make predictions
    sentiment = model.predict_classes(sequence)[0][0]
    return sentiment

# Streamlit app
def main():
    st.title("Movie Sentiment Analysis")
    st.write("Enter a movie review to predict the sentiment (positive or negative).")
    
    # Text input box
    review = st.text_area("Enter a movie review:", "")
    
    if st.button("Analyze"):
        if review:
            # Predict sentiment
            sentiment = predict_sentiment(review)
            
            # Display the sentiment
            if sentiment == 1:
                st.success("Positive sentiment")
            else:
                st.error("Negative sentiment")
        else:
            st.warning("Please enter a movie review.")
            
if __name__ == '__main__':
    main()
