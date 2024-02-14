import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():

    # Reading in preprocessed data into a dataframe
    dataframe = pd.read_csv("../Data/tweets_data1_preprocessed.csv")

    # Creating a sample of the preprocessed data
    dataset = dataframe.sample(n=500000, random_state=42)

    # Splitting the data in to feature and target variables
    x_tweets = dataset['tweet']
    y_labels = dataset['label']

    # Splitting the data into train and test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_tweets, y_labels, test_size=0.25, random_state=42)

    # Creating the TFIDF vectoriser and fitting the model with the feature data
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 1), max_df=0.05, min_df=1)
    x_train_tfidf_fit_transform = tfidf_vect.fit_transform(x_train)
    x_test_tfidf_transform = tfidf_vect.transform(x_test)

    # Saving the TFIDF model
    joblib.dump(tfidf_vect, '../Models/tfidfModel.pkl')

    # Creating an SVD model and reducing the features
    svd = TruncatedSVD(1110)
    x_train = svd.fit_transform(x_train_tfidf_fit_transform)
    x_test = svd.transform(x_test_tfidf_transform)

    # Saving SVD model
    joblib.dump(svd, '../Models/svdModel.pkl')

    # Splitting the training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=42)

    # Fitting the Logistic Regression model
    lr = LogisticRegression(max_iter=3000)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_val)

    # Getting the accuracy and confusion matrix for the model
    accuracy = accuracy_score(y_val, y_pred)
    confusion_mat = confusion_matrix(y_val, y_pred, labels=lr.classes_)
    confusion_matrix_displayed = ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat, display_labels=lr.classes_)
    confusion_matrix_displayed.plot()

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_mat)
    print(plt.show())

    # Saving the logistic regression model
    joblib.dump(lr, '../Models/LogRegModel.pkl')


if __name__ == "__main__":
    main()
