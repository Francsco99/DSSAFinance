#Description: This program attempts to optimize a users portfolio using the Efficient Frontier & Python

#Import the python libraries

#pandas is used to retrieve data from yahoo finance
from pandas_datareader import data as web
import pandas as pd

#yahoo finance
import yfinance as yfin
yfin.pdr_override()

import numpy as np

#random library to generate weights values
import random

#matplotlib to plot the data
import matplotlib.pyplot as plt
plt.style.use('classic')

from pypfopt.efficient_frontier import efficient_frontier
from pypfopt import risk_models
from pypfopt import expected_returns

from pypfopt.efficient_frontier.efficient_semivariance import EfficientFrontier

#Get the discrete allocation of each share per stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

#Portfolio definition, first 100 stocks of SP500 index

#removed BKNG for plot legibility
#Added FRT
sp500_first100 = ['AAPL','MSFT','AMZN','GOOGL','BRK-B','GOOG','XOM','UNH','JNJ','NVDA','JPM','V','TSLA','PG','CVX','HD','MA','META',
                    'MRK','LLY','ABBV','PFE','PEP','KO','BAC','TMO','AVGO','COST','WMT','ABT','MCD','CSCO','DIS','DHR','ACN','VZ','CMCSA',
                    'NEE','WFC','LIN','ADBE','PM','NKE','BMY','TXN','COP','CRM','NFLX','AMGN','RTX','T','HON','ORCL','QCOM','UPS','CAT','IBM',
                    'UNP','LOW','MS','SPGI','SBUX','GS','INTC','SCHW','BA','CVS','ELV','DE','PLD','BLK','AMD','INTU','MDT','GILD','AMT','LMT',
                    'ADP','C','CI','TJX','AMAT','CB','ISRG','AXP','PYPL','MDLZ','TMUS','NOW','SYK','ADI','MMC','GE','SLB','MO','VRTX','DUK','EOG',
                    'REGN','FRT']

#Portfolio definition, i choose the stocks
my_portfolio = ['GC=F','NVR','BKNG','AZO','CMG','IMMR']

#Generate a array of random weights for the stokcs in the portfolio my_portfolio

num_stocks = len(my_portfolio) #choose here the portfolio

# Generate a array of 50 random numbers between 0 and 1
weights = np.array([random.uniform(0,1) for _ in range(num_stocks)])

# Scale the numbers so that their sum is equal to 1
s = np.sum(weights)
weights = np.round(weights/s, 5)

print(weights)

#Or assing custom weights

#weights = np.array([0.1, 0.1])

#Get the portfolio starting date
stockStartDate = '2018-01-01'

#Get the portfolio ending date
stockEndDate = '2020-01-01'

#Get the Adjusted Close information for each stock in the portfolio
df = web.get_data_yahoo(my_portfolio, stockStartDate,stockEndDate)['Adj Close']

#Show the values of the data frame
df

#Visually show the portfolio
title = 'Portfolio Close Price History'

#Get the stocks
my_stocks = df

#Create and plot the graph
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c], label = c)
plt.title(title)
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
#plt.legend(my_stocks.columns.values, loc = 'upper right')

#Download plot
#plt.savefig("fig.pdf")
#from google.colab import files
#files.download("fig.pdf")

#Show the plot
plt.show()

#Calculate the daily simple return = stock_price/stock_price_precedent -1
returns = df.pct_change()

#Show the daily simple return
returns

#Create and show the annualized covariance matrix
cov_matrix_annual = returns.cov() * 252 #252 trading days in a year

#Show the matrix
cov_matrix_annual

#Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual,weights))
port_variance

#Calculate the portfolio volatility aka standard deviation
port_volatility = np.sqrt(port_variance)
port_volatility

#Calculate the annual portfolio return
portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights)*252
portfolioSimpleAnnualReturn

#Show the expected annual return, volatility (risk), and variance

percent_var = str(round(port_variance,2)*100)+'%'
percent_vols = str(round(port_volatility,2)*100)+'%'
percent_ret = str(round(portfolioSimpleAnnualReturn,2)*100)+'%'

print('Expected annual return: '+ percent_ret)
print('Annual volatility (risk): '+percent_vols)
print('Annual variance: '+percent_var)

#Portfolio Optimization!

#Calculate the expected returns and the annualised sample covariance matrix of asset returns

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#Optimize for maximal Sharpe Ratio
ef = EfficientFrontier(mu,S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights,latest_prices, total_portfolio_value=15000)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation: ',allocation)
print('Funds remaining: ${:.2f}'.format(leftover))