import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    categorical_columns = [
        'gender',
        'married',
        'dependents',
        'self_employed'
    ]

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        df['loanamount'] = df['loanamount'].fillna(df['loanamount'].median())
        df['loan_amount_term'] = df['loan_amount_term'].fillna(df['loan_amount_term'].median())
        df['credit_history'] = df['credit_history'].fillna(df['credit_history'].median())

    le = LabelEncoder()

    categorical_features = [
        'gender',
        'married',
        'dependents',
        'education',
        'self_employed',
        'property_area',
        'loan_status'
    ]

    for col in categorical_features:
        df[col] = le.fit_transform(df[col])

    X = df.drop(
        ['loan_id', 'loan_status'],
        axis=1
    )

    y = df['loan_status']

    print("\nMissing values:")
    print(df.isnull().sum())

    return X, y, df
