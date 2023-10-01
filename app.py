import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the trained model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Function to preprocess the input text
def preprocess_text(text):
    # Tokenize and encode the text
    encoding = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    return encoding

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    encoding = preprocess_text(text)
    # Make predictions
    with torch.no_grad():
        logits = model(**encoding)[0]
        probabilities = torch.softmax(logits, dim=1).squeeze()
    sentiment = probabilities[1].item()  # Probability of positive sentiment
    return sentiment

# Streamlit app
def main():
    st.title("Movie Sentiment Analysis")

   # Add a picture
    st.image("barbie.jpg", caption="Barbie (movietitle)", use_column_width=True, width=34, height=34)

    
    st.write("Enter a movie review to predict the sentiment (positive or negative).")

    # Text input box
    review = st.text_area("Enter a movie review:", "")

    if st.button("Analyze"):
        if review:
            # Predict sentiment
            sentiment = predict_sentiment(review)

            # Display the sentiment
            if sentiment >= 0.5:
                st.success("Positive sentiment")
            else:
                st.error("Negative sentiment")
        else:
            st.warning("Please enter a movie review.")

if __name__ == '__main__':
    main()
