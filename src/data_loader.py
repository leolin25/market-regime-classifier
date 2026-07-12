import yfinance as yf
import pandas as pd

def load_data(ticker: str, start_date: str, end_date: str):
    print(f'Downloading data for {ticker} from {start_date} to {end_date}')
    df = yf.download(ticker, start=start_date, end=end_date)

    # We only need close for our model
    df = df[['Close']].copy()
    df.rename(columns={'Close': 'Price'}, inplace=True)

    return df