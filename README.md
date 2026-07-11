# Market Regime Classification Engine

## Overview

This project is a machine learning pipeline which takes in financial data from a specified period of time and yfinance library, it classifies the historical market data into distinct 'regimes' (Low/Medium/High volatility). By identifying these market states we can dynamically change the approach around the market as risk is changing.

The engine calculates statistical features of the data and uses a random forest classification to identify the shifts in market behaviour. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

## Libraries used

- yfinance
- pandas, numpy
- matplotlib
- scikit-learn

## How to run the project

1. Clone repository

```
git clone [https://github.com/leolin25/market-regime-classifier.git](https://github.com/leolin25/market-regime-classifier.git)
cd market-regime-classifier
```

2. Setup the virual environment (Windows)

```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Run the engine

```
python main.py
```

## Important Files

- data_exploration.ipynb
Data visualisation and exploration done in this notebook. It justifies some of the steps.

- data_loader.py
yfinance API connections

- features.py
Log returns and rolling statistics calculations

- model.py
Random forest model architecture

- visualise.py
Produces the graphs using matplotlib

## Example

This is an example of regime classification done on SPY from 2016-2026. This is a graph produced in the data exploration notebook

![Regime Classification Chart](outputs/regime_chart.png)