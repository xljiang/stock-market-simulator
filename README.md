# stock-market-simulator

### Description

This project implements a stock market simulator. The goal is to let the simulator generate the best trading strategy automatically and tell which stock we should buy or sell to make the most benefit.

The project contains three different major components:

- A Market Simulator: Given a portfolio (which is like a personal account that has information about what stocks you are currently holding) and an order book (a series of BUY/SELL orders), return the estimated result. The market simulator 
- Machine Learning Models: In this project, the stock price is predicted from historical price data. I applied a few machine learning models such as linear regression, random forest, and Q learning to train models based on historical data. For the learning purpose, I generated these machine learning models from scratch other than use pre-defined libraries.
- Trading Strategy Generator: Apply the trained learning models and convert it to a classification learner using different technical indicators (e.g. Moving Average, Bollinger Band, Relative Strength Index and so on). The classification learners are then used to determine a series of BUY/SELL orders to form the best trading strategy.

### Technologies

**Python**: NumPy, Pandas, SciPy, Matplotlib   
**Machine Learning**: Linear Regression, Random Forest, Q learning with Dyna feature
