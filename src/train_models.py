from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_logistic_regression(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model