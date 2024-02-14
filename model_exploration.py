import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from Classification import Classification
from sklearn.decomposition import TruncatedSVD


def main():
    # Creating a dataframe with preprocesed data
    dataframe = pd.read_csv("../Data/tweets_data1_preprocessed_sampled.csv")
    dataset = dataframe.sample(n=10000, random_state=42)

    # Separating x and y
    x_tweets = dataset['tweet']
    y_labels = dataset['label']

    x_train, x_test, y_train, y_test = train_test_split(
        x_tweets, y_labels, test_size=0.1, random_state=42)

    # Min_df is something you can adjust potentially, same with ngram_range
    tfidf_vect = TfidfVectorizer(ngram_range=(1, 1), max_df=0.05, min_df=1)

    x_train_tfidf_fit_transform = tfidf_vect.fit_transform(x_train)

    x_test_tfidf_transform = tfidf_vect.transform(x_test)

    svd = TruncatedSVD(50)

    x_train = svd.fit_transform(x_train_tfidf_fit_transform)
    x_test = svd.fit_transform(x_test_tfidf_transform)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=.25, random_state=42)

    # SVM

    svm = Classification('SVM', x_train, x_val, y_train, y_val)

    param_grid = {'C': [0.12, 0.14, 0.16],
                  'kernel': ['linear']}

    print(svm.get_scores(param_grid))

    param_grid = {'kernel': ['poly'],
                  'degree': [0.5, 1, 1.5],
                  'gamma': ['scale', 'auto']}
    print(svm.get_scores(param_grid))

    # Decision tree

    dec_tree = Classification('Decision Tree', x_train, x_val, y_train, y_val)

    param_grid = {'min_samples_leaf': [28, 29, 30, 31, 32],
                  'max_depth': [18, 20, 22, 24, 26]}

    print(dec_tree.get_scores(param_grid))

    param_grid = {'min_samples_leaf': [28, 29, 30, 31, 32],
                  'max_depth': [16, 17, 18, 19, 20]}
    print(dec_tree.get_scores(param_grid))

    param_grid = {'min_samples_leaf': [30, 40, 50, 60, 70, 80, 90, 100],
                  'max_depth': [1, 2, 3, 4, 5, 6, 7, 8]}
    print(dec_tree.get_scores(param_grid))

    # Random forest

    random_forest = Classification(
        'Random Forest', x_train, x_val, y_train, y_val)

    param_grid = {'min_samples_leaf': [11, 12, 13, 14],
                  'max_depth': [75, 100, 125, 150]}

    print(random_forest.get_scores(param_grid))

    param_grid = {'min_samples_leaf': [8, 9, 10, 11, 12, 13, 14],
                  'max_depth': [20, 25, 30, 35, 40, 45, 50]}
    print(random_forest.get_scores(param_grid))

    param_grid = {'min_samples_leaf': [8, 9, 10, 11, 12, 13, 14],
                  'max_depth': [5, 10, 15, 20, 25]}
    print(random_forest.get_scores(param_grid))

    # Guassian Naive Bayes

    gnb = Classification('Naive Bayes', x_train, x_val, y_train, y_val)

    param_grid = {'var_smoothing': [1e-8, 1e-4, 1e-2, 1e-1, 0.5, 1]}

    print(gnb.get_scores(param_grid))

    param_grid = {'var_smoothing': [1, 1.5, 2, 2.5]}

    print(gnb.get_scores(param_grid))

    # Logistic Regression

    log_reg = Classification('Logistic Regression',
                             x_train, x_val, y_train, y_val)

    param_grid = {'penalty': ['l2'],
                  'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}

    print(log_reg.get_scores(param_grid))

    param_grid = {'penalty': ['l2'],
                  'C': [1.25, 1.5, 1.75, 2]}
    print(log_reg.get_scores(param_grid))

    param_grid = {'penalty': ['l2', 'none'],
                  'C': [0.001, 1, 1000],
                  'solver': ['sag', 'lbfgs', 'newton-cg'],
                  'max_iter': [3000]}
    print(log_reg.get_scores(param_grid))


if __name__ == "__main__":
    main()
