import os
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.features import add_features
from src.model import classification

# creates a visualisation for the prediction and saves in output folder
def plot_regimes(df, ticker):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['Price'])

    # color-code the background based on regime
    colors = {0: 'green', 1: 'yellow', 2: 'red'}
    label_added = {0: False, 1: False, 2: False} # to ensure we dont add a label to the legend repeatedly

    for i in range(1, len(df)):
        current_regime = df['target_regime'].iloc[i]

        # add a label once per regime
        label = f'Regime {current_regime}' if not label_added[current_regime] else ''
        if not label_added[current_regime]:
            label_added[current_regime] = True

        ax.axvspan(df.index[i-1], df.index[i], color=colors[current_regime], label=label, alpha=0.2, lw=0)

    # format graph
    ax.set_title(f'Market Regime Classfication using Random Forest ({ticker})')
    ax.set_ylabel('Price (USD)')
    ax.set_xlabel('Date')
    ax.margins(x=0)

    # add a custom legend
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    ax.legend([legend_dict.get(f'{ticker} Price'), legend_dict.get('Regime 0'), legend_dict.get('Regime 1'), legend_dict.get('Regime 2')],
            ['Price', 'Low Volatility', 'Medium Volatility', 'High Volatility'], 
            loc='upper left')

    plt.tight_layout()

    # save fig in outputs folder
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/regime_chart.png', dpi=300)
    print("Saved graph to outputs/regime_chart.png")

# change these variables to get different results
if __name__ == "__main__":
    TICKER = "SPY"
    START_DATE = "2016-01-01"
    END_DATE = "2025-12-31"
    WINDOW = 20
    
    df = load_data(TICKER, START_DATE, END_DATE)
    df = add_features(df, window=WINDOW)
    df, model = classification(df)

    plot_regimes(df, TICKER)