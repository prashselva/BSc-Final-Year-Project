import joblib
import re
import nltk
import string
import joblib
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD


class SentimentAnalyser:

    global stopwords
    additionalStopwords = ['rt', 'rts', 'retweet']
    stopwords = set().union(stopwords.words('english'), additionalStopwords)

    def preprocessing(self, text):

        # Making all tweets lowercase
        text_lower = text.lower()
        # Removing URLs (change code)
        processed_text = re.sub(
            '((www.[^s]+)|(https?://[^s]+))', ' ', str(text_lower))

        #Removing @name
        processed_text = re.sub(r"@\w+\b", ' ', processed_text)

        # Removing stopwords
        processed_text = " ".join([word for word in str(
            processed_text).split() if word not in stopwords])

        # Removing punctuation
        puncs_to_remove = string.punctuation
        processed_text = processed_text.translate(
            str.maketrans('', '', puncs_to_remove))

        # Removing repeated characters (change code)
        processed_text = re.sub(r'([a-z])\1+', r'\1', processed_text)

        # Removing numbers (change code)
        processed_text = re.sub('[0-9]+', '', processed_text)

        # Tokenising
        processed_text = processed_text.split()

        # Stemming
        processed_text = [nltk.PorterStemmer().stem(word)
                          for word in processed_text]

        return processed_text

    def tfidfVectoriser(self, processed_text):

        # load the tfidf vectoriser
        testingvectoriser = joblib.load('../Models/tfidfModel.pkl')

        # Vectorising
        vectorised_text = testingvectoriser.transform(processed_text)

        testingsvd = joblib.load('../Models/svdModel.pkl')

        vectorised_text_df = testingsvd.transform(vectorised_text)

        return vectorised_text_df

    def analyse_sentiment(self, text):

        processed_text = self.preprocessing(text)

        vectorised_text_df = self.tfidfVectoriser(processed_text)

        # Load the saved model from the pickle file
        model = joblib.load('../Models/LogRegModel.pkl')

        # Model Prediction
        predictions = model.predict(vectorised_text_df)

        if predictions[0] == 4:
            return "Positive"
        else:
            return "Negative"
