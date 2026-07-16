from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# helper function which classifies the market regime
def assign_regime(x, low_vol, high_vol):
    if x <= low_vol:
        return 0
    elif x <= high_vol:
        return 1
    else:
        return 2

# train a model to classify the market regimes and use it to predict
def classification(df: pd.DataFrame):
    low_vol = df['volatility'].quantile(0.33)
    high_vol = df['volatility'].quantile(0.66)

    # define features and targets
    df['target_regime'] = df['volatility'].apply(lambda x: assign_regime(x, low_vol, high_vol))
    features = ['rolling_avg_price', 'volatility']
    X = df[features]
    y = df['target_regime']

    # split the data into training and test data
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f'{len(X_train)} Training Samples, {len(X_test)} Test Samples')

    # train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    print(f'Accuracy of model on test data: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['Low Vol', 'Med Vol', 'High Vol']))

    df['predicted_regime'] = model.predict(X)

    return df, model