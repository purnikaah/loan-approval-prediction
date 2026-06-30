from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def tune_decision_tree(X_train, y_train):

    param_grid = {
        'max_depth': [1,2,3,4,5,6,7,8,9,10],
        'criterion': ['gini', 'entropy']
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)

    print("Best DT Parameters:",
          grid.best_params_)

    print("Best DT Accuracy:",
          grid.best_score_)

    return grid.best_estimator_


def tune_random_forest(X_train, y_train):

    param_grid = {
        'n_estimators': [50,100,150],
        'max_depth': [3,5,7,10]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)

    print("Best RF Parameters:",
          grid.best_params_)

    print("Best RF Accuracy:",
          grid.best_score_)

    return grid.best_estimator_


def tune_logistic_regression(X_train, y_train):

    param_grid = {
        'C': [0.01,0.1,1,10,100]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)

    print("Best LR Parameters:",
          grid.best_params_)

    print("Best LR Accuracy:",
          grid.best_score_)

    return grid.best_estimator_