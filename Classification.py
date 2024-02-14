import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold


class Classification():

    """
    This class creates a classification algorithm out of Logistic Regression, Decision Tree, Random Forest, and SVM.

    Parameters:

        model_type: 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'
        the type of classifcation algorithm you would like to apply 

        x_train: dataframe
        the independant variables of the training data

        x_val: dataframe
        the independant variables of the validation data

        y_train: series
        the target variable of the training data

        y_val: series
        the target variable of the validation data

    """

    def __init__(self, model_type, x_train, x_val, y_train, y_val):

        self.model_type = model_type
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        if self.model_type == 'Logistic Regression':
            self.classifier = LogisticRegression(fit_intercept=False)
        elif self.model_type == 'Decision Tree':
            self.classifier = DecisionTreeClassifier(random_state=42)
        elif self.model_type == 'Random Forest':
            self.classifier = RandomForestClassifier(
                n_estimators=20, n_jobs=-1, random_state=42)
        elif self.model_type == 'SVM':
            self.classifier = SVC()
        elif self.model_type == 'Naive Bayes':
            self.classifier = GaussianNB()
        elif self.model_type == 'KNN':
            self.classifier = KNeighborsClassifier(n_jobs=-1)

    def get_scores(self, param_grid):
        """_summary_

        Performs a gridsearch cross validation with given hyperparameters and data.
        Gets the accuracy for the given data and creates a dataframe containing scores.

        Parameters:
            param_grid (dictionary): parameters for chosen classification algorithm to be passed through gridsearch cross validation

        Returns:
            classification_report_pd: dataframe consisting of the classification accuracy and validation results
        """

        fit_classifier = self.classifier.fit(self.x_train, self.y_train)

        opt_model = GridSearchCV(fit_classifier,
                                 param_grid,
                                 cv=StratifiedKFold(
                                     n_splits=5, random_state=42, shuffle=True),
                                 scoring='accuracy',
                                 return_train_score=True,
                                 n_jobs=-1)

        opt_model = opt_model.fit(self.x_train, self.y_train)

        self.best_model = opt_model.best_estimator_

        self.acc_train = self.best_model.score(self.x_train, self.y_train)

        self.acc_val = self.best_model.score(self.x_val, self.y_val)

        d = {'Model Name': [self.model_type],
             'Train Accuracy': [self.acc_train],
             'Validation Accuracy': [self.acc_val],
             'Accuracy Difference': [self.acc_train-self.acc_val]}

        # A dataframe with the model used, the train accuracy and validation accuracy
        scores_table = pd.DataFrame(data=d)

        print(scores_table)

        best_params = opt_model.best_params_

        if param_grid == {}:
            pass
        else:
            print("The best hyperparameters are: ", best_params, '\n')

        self.y_validated = self.best_model.predict(self.x_val)

        classification_report_pd = pd.DataFrame.from_dict(classification_report(
            self.y_val, self.y_validated, output_dict=True)).iloc[0:3, 0:5]

        return classification_report_pd
