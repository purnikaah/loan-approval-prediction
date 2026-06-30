from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_and_preprocess_data
from src.train_models import (
    train_decision_tree,
    train_random_forest,
    train_logistic_regression
)
from src.evaluate_models import evaluate_model

X, y, df = load_and_preprocess_data(
    "data/dataset.csv"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

dt_model = train_decision_tree(
    X_train,
    y_train
)

rf_model = train_random_forest(
    X_train,
    y_train
)

lr_model = train_logistic_regression(
    X_train,
    y_train
)

print("\nDecision Tree Results")
evaluate_model(
    dt_model,
    X_test,
    y_test
)

print("\nRandom Forest Results")
evaluate_model(
    rf_model,
    X_test,
    y_test
)

print("\nLogistic Regression Results")
evaluate_model(
    lr_model,
    X_test,
    y_test
)