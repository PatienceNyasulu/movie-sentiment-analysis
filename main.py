import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer



# Load the tokenizer
#with open('tokenizer.pickle', 'rb') as handle:
    #tokenizer = pickle.load(handle)

from transformers import BertTokenizerFast, TFBertForSequenceClassification

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Load the trained model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')


# Load the trained model
#model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Function to preprocess the input text
#def preprocess_text(text):
    # Tokenize the text
  #  sequence = tokenizer.texts_to_sequences([text])
    # Pad the sequence
  #  sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
  #  return sequence

def preprocess_text(text):
    # Tokenize and encode the text
    encoding = tokenizer.encode_plus(text, max_length=100, truncation=True, padding='max_length', return_tensors='tf')
    return encoding

# Function to predict the sentiment
#def predict_sentiment(text):
    # Preprocess the text
  #  sequence = preprocess_text(text)
    # Make predictions
#    sentiment = model.predict_classes(sequence)[0][0]
 #   return sentiment

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    encoding = preprocess_text(text)
    # Make predictions
    prediction = model.predict(encoding)
    sentiment = prediction[0][0]
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
