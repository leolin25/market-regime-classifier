import streamlit as st
import matplotlib.pyplot as plt
from datetime import date, timedelta
import os

# Import your existing engine logic
from src.data_loader import load_data
from src.features import add_features
from src.model import classification

# page configuration
st.set_page_config(page_title="Market Regime Classifier", layout="wide")
st.title('Market Regime Classification Engine')
st.markdown('Using machine learning to classify market regimes based on historical data')

# sidebar for user parameters
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input('Enter Stock Ticker', value='SPY').upper()
start_date = st.sidebar.date_input('Start Date', value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input('End Date', value=date.today())

# hyperparameter selection
window_size = st.sidebar.slider('Window Size (dats)', min_value=5, max_value=50, value=20, step=1)

run_button = st.sidebar.button('Run model')

if run_button:
    try:
        df = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        df = add_features(df, window_size)
        df = classification(df)

        # display metrics
        st.subheader('Model Performance Metrics')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Total Trading Days', len(df))
        col2.metric('Low Volatility Days', len(df[df['regime'] == 'Low Volatility']))
        col3.metric('High Volatility Days', len(df[df['regime'] == 'High Volatility']))
        col4.metric('Medium Volatility Days', len(df[df['regime'] == 'Medium Volatility']))

        # generate the chart
        st.subheader(f'{ticker} Historical  Market Regimes')
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df.index, df['Price'])
        colors = {0: 'green', 1: 'yellow', 2: 'red'}
        label_added = {0: False, 1: False, 2: False}

        # color all regimes on the chart
        for i in range(1, len(df)):
            current_regime = df['regime'].iloc[i]
            label = f'Regime {current_regime}' if not label_added[current_regime] else ''

            if not label_added[current_regime]:
                label_added[current_regime] = True

            ax.axvspan(df.index[i-1], df.index[i], color=colors[current_regime], label=label, alpha=0.2, lw=0)

        # format graph
        ax.set_title(f'Market Regime Classfication using Random Forest ({ticker})')
        ax.set_ylabel('Price (USD)')
        ax.set_xlabel('Date')

        # add a custom legend
        handles, labels = ax.get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))
        ax.legend([legend_dict.get('SPY Price'), legend_dict.get('Regime 0'), legend_dict.get('Regime 1'), legend_dict.get('Regime 2')],
                ['SPY Price', 'Low Volatility', 'Medium Volatility', 'High Volatility'], 
                loc='upper left')
        
        # save fig in outputs folder
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/regime_chart.png', dpi=300)

        # render the plot in Streamlit
        st.pyplot(fig)

        # show the raw data table (last 100 rows)
        with st.expander("View Raw Data"):
            st.dataframe(df.tail(100))

    except Exception as e:
            st.error(f"An error occurred: {e}. Please check if the ticker symbol is valid.")