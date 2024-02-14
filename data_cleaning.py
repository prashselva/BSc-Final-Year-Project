import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords


def main():

    # Loading the dataset, encoding so incompatible values are removed (utf-8), no headers so setting my own column titles
    dataset = pd.read_csv("../Data/tweets_data1.csv", encoding='latin-1', header=None,
                          names=['label', 'idno', 'datetime', 'NO_QUERY', 'username', 'tweet'])

    # Head of the dataset
    print(dataset.head())

    # Shape of the dataset
    print(dataset.shape)

    # Dropping duplicate tweets
    dataset.drop_duplicates(subset='tweet', inplace=True)

    # Dropping unnecessary columns as they are not needed
    dataset = dataset.drop(dataset.columns[[1, 2, 3, 4]], axis=1)

    # Checking if balanced
    print(dataset.groupby('label').count())

    # New head of the dataset
    print(dataset.head)

    # Checking if any values are null (they are not)
    dataset.isnull().any()

    # Graph showing the number of negative and positive tweets, both are about the same
    graph = dataset.groupby('label').count().plot(kind='bar')

    print(plt.show())

    # Number of positive and negative tweets
    print(dataset.groupby('label').count())

    # Balancing number of positive and negative tweets

    negative_len = len(dataset[dataset['label'] == 0])
    # print(negative_len)

    tweet_negative_class_indices = dataset[dataset['label'] == 0].index
    tweet_positive_class_indices = dataset[dataset['label'] == 4].index

    random_positive_indices = np.random.choice(
        tweet_positive_class_indices, negative_len, replace=False)
    # print(len(random_positive_indices))

    balanced_indices = np.concatenate(
        [tweet_negative_class_indices, random_positive_indices])
    dataset = dataset.loc[balanced_indices]

    # Checking if balanced
    print(dataset.groupby('label').count())

    # Head of the dataset
    print(dataset.head())

    # Shape of the dataset
    print(dataset.shape)

    dataset.to_csv("../Data/tweets_data1_balanced_simplified.csv", index=False)


if __name__ == "__main__":
    main()
