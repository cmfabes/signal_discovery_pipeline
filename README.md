
# Market Signal Demo Project

This project demonstrates a simple signal analysis workflow using historical stock price data. It shows how to take raw time-series market data, engineer basic features, flag unusual movements, and summarize findings in a way that supports business or investment decisions.

The focus is on clarity and process rather than complexity. The project is designed to highlight practical skills relevant to business and data analyst roles, including working with real data, defining simple rules for anomaly detection, and communicating insights.

---

##  What This Project Does

- Downloads daily price data for a selected stock symbol
- Calculates daily returns and rolling averages
- Computes rolling volatility (standard deviation of returns)
- Flags "anomaly days" where returns exceed a defined threshold
- Outputs a CSV file and prints a short summary of what was detected

---

## Skills Demonstrated

- Working with time-series financial data
- Data cleaning and transformation using Python and pandas
- Simple feature engineering (returns, rolling metrics, volatility)
- Basic anomaly detection logic
- Translating numerical patterns into plain-language observations

---

##  Tech Stack

- Python 3.9+
- pandas
- yfinance (for price data)

---

## How to Run

1. Install dependencies:

```bash
pip install pandas yfinance
