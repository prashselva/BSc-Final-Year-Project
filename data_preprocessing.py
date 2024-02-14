import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# NLTK Stopwords
additionalStopwords = ['rt', 'rts', 'retweet']
stopwords = set().union(stopwords.words('english'), additionalStopwords)

# Removing URLs


def URL_removal(dataset):
    return re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', dataset)

# Removing @name


def at_name_removal(dataset):
    return re.sub(r"@\w+\b", ' ', dataset)

# Removing stop words


def stopword_removal(tweet):
    return " ".join([word for word in str(tweet).split() if word not in stopwords])

# Removing punctuation


def punctuation_removal(tweet, puncs_to_remove):
    return tweet.translate(str.maketrans('', '', puncs_to_remove))

# Removing repeated characters


def repeat_removal(tweet):
    return re.sub(r'([a-z])\1+', r'\1', tweet)

# Removing numbers


def numb_removal(dataset):
    return re.sub('[0-9]+', '', dataset)

# Stemming the data to simplify the words into root form


def stemming(data):
    stemmer = nltk.PorterStemmer()
    data = [stemmer.stem(word) for word in data]
    return data


def main():

    # Importing cleaned data
    dataset = pd.read_csv("../Data/tweets_data1_balanced_simplified.csv")

    # Preprocessing data
    dataset['tweet'] = dataset['tweet'].apply(lambda x: URL_removal(x))
    dataset['tweet'] = dataset['tweet'].apply(lambda x: at_name_removal(x))
    dataset['tweet'] = dataset['tweet'].apply(
        lambda text: stopword_removal(text))
    dataset['tweet'] = dataset['tweet'].apply(
        lambda x: punctuation_removal(x, string.punctuation))
    dataset['tweet'] = dataset['tweet'].apply(lambda x: repeat_removal(x))
    dataset['tweet'] = dataset['tweet'].apply(lambda x: numb_removal(x))

    # Tokenising
    dataset['tweet'] = dataset['tweet'].apply(RegexpTokenizer(r'\w+').tokenize)

    # Stemming tokenised data
    dataset['tweet'] = dataset['tweet'].apply(lambda x: stemming(x))

    # Taking a random sample of the dataset to run on a gridsearch to find the most ideal hyperparameters
    sampled_dataset = dataset.sample(frac=0.2, random_state=42, replace=False)

    sampled_dataset.to_csv(
        "../Data/tweets_data1_preprocessed_sampled.csv", index=False)
    dataset.to_csv("../Data/tweets_data1_preprocessed.csv", index=False)


if __name__ == "__main__":
    main()
