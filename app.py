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
#for calculating percentage sentiment score
from textblob import TextBlob
def calculate_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Streamlit app
def main():
    st.title("Movie Sentiment Analysis using Bert")

   # Add a picture
    st.image("barbie.jpg", caption="Barbie (movietitle)", use_column_width=True)

    
    st.write("Enter a movie review for the Barbie movie to predict the sentiment (positive or negative).")

    # Text input box
    review = st.text_area("Enter a review :", "")

    if st.button("Analyze"):
        sentiment_score = calculate_sentiment(review)
        # Display sentiment score
        st.write("Sentiment Score:", sentiment_score)
        # Interpret sentiment
        if sentiment_score > 0:
            st.write("Sentiment Score: Positive")
        else :
            st.write("Sentiment Score: Negative")
       
    if review:
            # Predict sentiment
        sentiment = predict_sentiment(review)

            # Display the sentiment
        if sentiment ==1:
             st.success("Positive sentiment")
        else:
             st.error("Negative sentiment")
    else:
        st.warning("Please enter a movie review.")

if __name__ == '__main__':
    main()
