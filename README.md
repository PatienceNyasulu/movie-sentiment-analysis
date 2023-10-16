
## This project was done by Praise T Ganyiwa r204436q & Gamuchirai P Nyasulu r205734y


This repository contains code for performing sentiment analysis on movie reviews using the BERT (Bidirectional Encoder Representations from Transformers) model and deploying it as a web application using Streamlit. The app predicts whether a given movie review has a positive or negative sentiment.

## Link to deployed Streamlit app

https://movie-sentiment-analysis-ehmn9tdga4smvhg6sfb7vt.streamlit.app/

## Installation

To run this application locally, you need to follow these steps:

1. Clone the repository:

   `````shell
   git clone https://github.com/your-username/movie-sentiment-analysis.git
   cd movie-sentiment-analysis
   ```

2. Install the required dependencies using pip:

   ````shell
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ````shell
   streamlit run app.py
   ```

   This will start the Streamlit development server, and you can access the app in your browser at `http://localhost:8501`.

## Usage

1. Once you access the app in your browser, you will see the title "Movie Sentiment Analysis using BERT" along with an image of a Barbie movie.

2. Enter a movie review in the text area provided.

3. Click the "Analyze" button to perform sentiment analysis on the entered review.

4. The app will display the sentiment score of the review, along with an interpretation of the sentiment as "Positive" or "Negative".

5. The sentiment score is calculated using the TextBlob library, which provides a polarity score ranging from -1 (most negative) to 1 (most positive). If the sentiment score is greater than 0, it is considered a positive sentiment; otherwise, it is considered a negative sentiment.

6. Additionally, the app uses the BERT model to predict the sentiment of the review. If the sentiment score is not exactly 0.5, it is interpreted as a negative sentiment; otherwise, it is interpreted as a positive sentiment.

7. If no movie review is entered, a warning message will be displayed, prompting the user to enter a review.



## License

This project is licensed under the MIT License. Feel free to use and modify the code as per your needs.
