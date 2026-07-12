import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame, window: int=20) -> pd.DataFrame:
    # add features needed for machine learning engine
    df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
    df['rolling_avg_price'] = df['Price'].rolling(window).mean()
    df['volatility'] = df['Price'].rolling(window).std() * np.sqrt(252)

    # drop null rows since they cause errors in random forest
    df.dropna(inplace=True)

    return df