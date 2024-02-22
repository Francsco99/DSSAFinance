#Import the python libraries

#pandas is used to retrieve data from yahoo finance
from pandas_datareader import data as web
import pandas as pd

#yahoo finance
import yfinance as yfin
yfin.pdr_override()

import numpy as np
import random

from google.colab import drive

#matplotlib to plot the data
import matplotlib.pyplot as plt
plt.style.use('classic')

import seaborn as sns

#scipy.optimize to run optimization
from scipy.optimize import minimize


#Import libraries
from pypfopt.efficient_frontier import efficient_frontier
from pypfopt import risk_models
from pypfopt import expected_returns

from pypfopt.efficient_frontier.efficient_semivariance import EfficientFrontier

"""#Getting data for the portfolios"""

#Mount google drive location
drive.mount('/content/drive')

#Load csv file
df_tickers = pd.read_csv('/content/drive/My Drive/sp500.csv')

#Show the first rows of the data frame
print(df_tickers.head())

#Extract the first column 'Symbol' names
data = df_tickers['Symbol'].values

#Number of portfolios
n=10

# Array to contain the portfolios
initial_portfolios = []

#Create 10 portfolios with 100 random stocks each
for _ in range(n):
    # Create a list of 100 random tickers from the imported file
    lista_random = list(random.sample(list(data), 100))
    # Append the list
    initial_portfolios.append(lista_random)

# Ordina ogni lista dell'array in ordine alfabetico
portfolios = []

for lista in initial_portfolios:
  sl = list(sorted(lista))
  portfolios.append(sl)

#Portfolio starting and ending dates for historical data
stockStartDate = '2017-01-01'
stockEndDate = '2019-01-01'

# Array to contain pandas acquired data
historicalDF = []

#Get data for each portfolio
for p in portfolios:
  #Get the Adjusted Close information for each stock in the portfolio
  data = web.get_data_yahoo(p, stockStartDate,stockEndDate)['Adj Close']
  #Append the portfolios
  historicalDF.append(data)

#Portfolio t0 investment date
initialStartInvestimentDate = '2019-01-01'
initialEndInvestimentDate = '2019-01-03'

startDF = []
for p in portfolios:
  startData = web.get_data_yahoo(p, initialStartInvestimentDate,initialEndInvestimentDate)['Adj Close']
  startDF.append(startData)

#Portfolio final investment date
finalStartInvestmentDate = '2022-01-01'
finalEndInvestmentDate = '2022-01-04'

finalDF = []
for p in portfolios:
  finalData = web.get_data_yahoo(p, finalStartInvestmentDate,finalEndInvestmentDate)['Adj Close']
  finalDF.append(finalData)

"""#Initial setting of the portfolios, random weights assigned for each stock."""

#Calculate random initial weights for each portfolio

#Initialize empty array to contain the ordered dict for each portfolio
initial_weights = []

# Itera su ogni portfolio
for portfolio in portfolios:
    # Genera pesi casuali per ogni stock
    weights = [random.uniform(0, 1) for _ in range(len(portfolio))]
    # Normalizza i pesi
    total_weight = sum(weights)
    weights = [weight/total_weight for weight in weights]
    # Crea un dizionario dei pesi per questo portfolio
    portfolio_dict = dict(zip(portfolio, weights))
    # Aggiungi il dizionario alla lista di dizionari dei pesi
    initial_weights.append(portfolio_dict)

# Capitale iniziale
capital = 10000

# Array vuoto per i dizionari delle quantità
portfolio_quantities = []

# Loop sui portfoli per creare i dizionari delle quantità
for i, portfolio in enumerate(portfolios):
    # Accedi ai pesi normalizzati per questo portfolio
    weights_dict = initial_weights[i]
    # Calcola le quantità di denaro da allocare per ogni stock
    amounts = [weights_dict[stock] * capital for stock in portfolio]
    # Crea un dizionario delle quantità per questo portfolio
    portfolio_dict = dict(zip(portfolio, amounts))
    # Aggiungi il dizionario alla lista di dizionari delle quantità
    portfolio_quantities.append(portfolio_dict)

#Array di dizionari con azione e prezzo di chiusura per ogni portfolio
close_prices_t0 = []

#Prendi ogni portfolio nel dataframe e convertilo in dizionario
for portfolio in startDF:
  prices = portfolio.to_dict()

  #Crea nuovo dizionario per ripulire i timestamp
  prices_new = {}
  for key in prices.keys():
    nv= list(prices[key].values())[0]
    prices_new[key] = nv
  close_prices_t0.append(prices_new)

# Array vuoto per i nuovi dizionari contenenti le azioni e il numero di azioni acquistabili
portfolio_stocks = []

# Loop sui portfoli per creare i nuovi dizionari
for i, portfolio_dict in enumerate(portfolio_quantities):
    # Crea un nuovo dizionario per questo portfolio
    new_dict = {}
    for stock, investment in portfolio_dict.items():
        # Accedi al prezzo di chiusura per questa azione
        closing_price = close_prices_t0[i][stock]
        # Calcola la quantità di azioni acquistabile in base all'investimento e il prezzo di chiusura
        quantity = (investment / closing_price)
        # Aggiungi l'azione e il numero di azioni acquistabili al dizionario
        new_dict[stock] = quantity
    # Aggiungi il nuovo dizionario alla lista di nuovi dizionari
    portfolio_stocks.append(new_dict)

"""# Calculate useful parameters"""

#Calculate simple return for each portfolio

#Initialize empty array
returns = []

for data in historicalDF:
  #Calculate the daily simple return = stock_price/stock_price_precedent -1
  r = data.pct_change()
  #Append the result
  returns.append(r)

#Calculate the annual covariance matrix for each portfolio

#Initalize empty array
cov_matrix_annual = []

for r in returns:
  #Create and show the annualized covariance matrix
  cov_matrix = r.cov() * 252 #252 trading days in a year
  #Append the result
  cov_matrix_annual.append(cov_matrix)

#Calculate variance for each portfolio

#Initialize empty array
variances = []

for i in range(n):
  current_weights = np.array(list(initial_weights[i].values()))
  current_cov_matrix = cov_matrix_annual[i]
  port_variance = np.dot(current_weights.T, np.dot(current_cov_matrix,current_weights))
  #Append the result
  variances.append(port_variance)

#Calculate each portfolio volatility (std deviation)
volatilities = []
for v in variances:
  port_volatility = np.sqrt(v)
  #Append the result
  volatilities.append(port_volatility)

#Calculate the annual portfolio return for each portfolio

#Initialize empty array
simpleAnnualReturns = []

for i in range(len(returns)):
  current_return = returns[i]
  current_weights = np.array(list(initial_weights[i].values()))
  portfolioSimpleAnnualReturn = np.sum(current_return.mean()*current_weights)*252
  #Append the result
  simpleAnnualReturns.append(portfolioSimpleAnnualReturn)

#Print the variance and volatility for each portfolio
for j in range(n):
  print(f"portfolio {j+1}")
  print("variance: {:.20f}".format(variances[j]).replace(".",","))
  print("volatility: {:.20f}".format(volatilities[j]).replace(".",","))
  print("annual return: {:.20f}".format(simpleAnnualReturns[j]).replace(".",","))
  print("\n")

"""#Calculate performances before optimization"""

#Calculate the expected annual return, volatility (risk), and variance for each portfolio
percent_variances = []
percent_volatilities = []
percent_returns = []

for i in range(n):
  current_variance = variances[i]
  current_vol = volatilities[i]
  current_ret = simpleAnnualReturns[i]

  percent_var = (current_variance)*100
  percent_vols = (current_vol)*100
  percent_ret = (current_ret)*100

  #Append the results
  percent_variances.append(percent_var)
  percent_volatilities.append(percent_vols)
  percent_returns.append(percent_ret)

"""#vediamo come va il portfolio senza ottimizzazione, al tempo finale gennaio 2022
spoiler, va bene
"""

#Array di dizionari con azione e prezzo di chiusura per ogni portfolio
close_prices_tf = []

#Prendi ogni portfolio nel dataframe e convertilo in dizionario
for portfolio in finalDF:
  prices = portfolio.to_dict()

  #Crea nuovo dizionario per ripulire i timestamp
  prices_new = {}
  for key in prices.keys():
    nv= list(prices[key].values())[0]
    prices_new[key] = nv
  close_prices_tf.append(prices_new)

# Array vuoto per i nuovi dizionari contenenti le azioni e il numero di azioni acquistabili
portfolio_quantities_tf = []

# Loop sui portfoli per creare i nuovi dizionari
for i, portfolio_dict in enumerate(portfolio_stocks):
    # Crea un nuovo dizionario per questo portfolio
    new_dict = {}
    for stock, investment in portfolio_dict.items():
        # Accedi al prezzo di chiusura per questa azione
        closing_price = close_prices_tf[i][stock]
        # Calcola la quantità di azioni acquistabile in base all'investimento e il prezzo di chiusura
        quantity = (investment * closing_price)
        # Aggiungi l'azione e il numero di azioni acquistabili al dizionario
        new_dict[stock] = quantity
    # Aggiungi il nuovo dizionario alla lista di nuovi dizionari
    portfolio_quantities_tf.append(new_dict)

for i in range(n):
  print(portfolios[i])
  #print(initial_weights[i])
  print("quantita iniziali")
  print(portfolio_quantities[i])
  print("quantita finali")
  print(portfolio_quantities_tf[i])
  print("prezzi iniziali")
  print(close_prices_t0[i])
  print("quantita finali")
  print(close_prices_tf[i])
  #print(portfolio_stocks[i])
  print("\n")

"""#Portfolio optimization using PyPortfolioOpt library"""

#Initialize empty arrays
frontiers_sharpe = []
cleaned_weights_sharpe = []

#Calculate the expected returns and the annualised sample covariance matrix of asset returns
for data in historicalDF:
  mu = expected_returns.mean_historical_return(data)
  S = risk_models.sample_cov(data)

  #Optimize for maximal Sharpe Ratio
  ef = EfficientFrontier(mu,S)
  weights = ef.max_sharpe()
  cw = ef.clean_weights()
  frontiers_sharpe.append(ef)
  cleaned_weights_sharpe.append(cw)

from pypfopt.efficient_frontier.efficient_semivariance import EfficientFrontier

#Initialize empty arrays
frontiers_vol = []
cleaned_weights_vol = []

#Calculate the expected returns and the annualised sample covariance matrix of asset returns
for data in historicalDF:
  mu = expected_returns.mean_historical_return(data)
  S = risk_models.sample_cov(data)

  #Optimize for maximal Sharpe Ratio
  ef = EfficientFrontier(mu,S)
  weights = ef.min_volatility()
  cw = ef.clean_weights()
  frontiers_vol.append(ef)
  cleaned_weights_vol.append(cw)

for i in range(n):
  print(f"Portfolio {i+1}")
  print("Pesi ottimizzazione sharpe")
  print(cleaned_weights_sharpe[i])
  print("Pesi ottimizzazione volatilità")
  print(cleaned_weights_vol[i])

"""#performance dopo 1 ottimizzazione"""

# Capitale iniziale
capital = 10000

# Array vuoto per i dizionari delle quantità
portfolio_quantities_opt1 = []

# Loop sui portfoli per creare i dizionari delle quantità
for i, portfolio in enumerate(portfolios):
    # Accedi ai pesi normalizzati per questo portfolio
    weights_dict = cleaned_weights_sharpe[i]
    # Calcola le quantità di denaro da allocare per ogni stock
    amounts = [weights_dict[stock] * capital for stock in portfolio]
    # Crea un dizionario delle quantità per questo portfolio
    portfolio_dict = dict(zip(portfolio, amounts))
    # Aggiungi il dizionario alla lista di dizionari delle quantità
    portfolio_quantities_opt1.append(portfolio_dict)

# Array vuoto per i nuovi dizionari contenenti le azioni e il numero di azioni acquistabili
portfolio_stocks_opt1 = []

# Loop sui portfoli per creare i nuovi dizionari
for i, portfolio_dict in enumerate(portfolio_quantities_opt1):
    # Crea un nuovo dizionario per questo portfolio
    new_dict = {}
    for stock, investment in portfolio_dict.items():
        # Accedi al prezzo di chiusura per questa azione
        closing_price = close_prices_t0[i][stock]
        # Calcola la quantità di azioni acquistabile in base all'investimento e il prezzo di chiusura
        quantity = (investment / closing_price)
        # Aggiungi l'azione e il numero di azioni acquistabili al dizionario
        new_dict[stock] = quantity
    # Aggiungi il nuovo dizionario alla lista di nuovi dizionari
    portfolio_stocks_opt1.append(new_dict)

# Array vuoto per i nuovi dizionari contenenti le azioni e il numero di azioni acquistabili
portfolio_quantities_tf_opt1 = []

# Loop sui portfoli per creare i nuovi portfolio_stocks_opt1
for i, portfolio_dict in enumerate(portfolio_stocks_opt1):
    # Crea un nuovo dizionario per questo portfolio
    new_dict = {}
    for stock, investment in portfolio_dict.items():
        # Accedi al prezzo di chiusura per questa azione
        closing_price = close_prices_tf[i][stock]
        # Calcola la quantità di azioni acquistabile in base all'investimento e il prezzo di chiusura
        quantity = (investment * closing_price)
        # Aggiungi l'azione e il numero di azioni acquistabili al dizionario
        new_dict[stock] = quantity
    # Aggiungi il nuovo dizionario alla lista di nuovi dizionari
    portfolio_quantities_tf_opt1.append(new_dict)

for i in range(n):
  print(portfolios[i])
  #print(initial_weights[i])
  print("quantita iniziali")
  print(portfolio_quantities_opt1[i])
  print("quantita finali")
  print(portfolio_quantities_tf_opt1[i])
  print("prezzi iniziali")
  print(close_prices_t0[i])
  print("prezzi finali")
  print(close_prices_tf[i])
  #print(portfolio_stocks[i])
  print("\n")

"""#portfolio dopo 2 ottimizzazione"""

# Capitale iniziale
capital = 10000

# Array vuoto per i dizionari delle quantità
portfolio_quantities_opt2 = []

# Loop sui portfoli per creare i dizionari delle quantità
for i, portfolio in enumerate(portfolios):
    # Accedi ai pesi normalizzati per questo portfolio
    weights_dict = cleaned_weights_vol[i]
    # Calcola le quantità di denaro da allocare per ogni stock
    amounts = [weights_dict[stock] * capital for stock in portfolio]
    # Crea un dizionario delle quantità per questo portfolio
    portfolio_dict = dict(zip(portfolio, amounts))
    # Aggiungi il dizionario alla lista di dizionari delle quantità
    portfolio_quantities_opt2.append(portfolio_dict)

# Array vuoto per i nuovi dizionari contenenti le azioni e il numero di azioni acquistabili
portfolio_stocks_opt2 = []

# Loop sui portfoli per creare i nuovi dizionari
for i, portfolio_dict in enumerate(portfolio_quantities_opt2):
    # Crea un nuovo dizionario per questo portfolio
    new_dict = {}
    for stock, investment in portfolio_dict.items():
        # Accedi al prezzo di chiusura per questa azione
        closing_price = close_prices_t0[i][stock]
        # Calcola la quantità di azioni acquistabile in base all'investimento e il prezzo di chiusura
        quantity = (investment / closing_price)
        # Aggiungi l'azione e il numero di azioni acquistabili al dizionario
        new_dict[stock] = quantity
    # Aggiungi il nuovo dizionario alla lista di nuovi dizionari
    portfolio_stocks_opt2.append(new_dict)

# Array vuoto per i nuovi dizionari contenenti le azioni e il numero di azioni acquistabili
portfolio_quantities_tf_opt2 = []

# Loop sui portfoli per creare i nuovi portfolio_stocks_opt1
for i, portfolio_dict in enumerate(portfolio_stocks_opt2):
    # Crea un nuovo dizionario per questo portfolio
    new_dict = {}
    for stock, investment in portfolio_dict.items():
        # Accedi al prezzo di chiusura per questa azione
        closing_price = close_prices_tf[i][stock]
        # Calcola la quantità di azioni acquistabile in base all'investimento e il prezzo di chiusura
        quantity = (investment * closing_price)
        # Aggiungi l'azione e il numero di azioni acquistabili al dizionario
        new_dict[stock] = quantity
    # Aggiungi il nuovo dizionario alla lista di nuovi dizionari
    portfolio_quantities_tf_opt2.append(new_dict)

"""#Portfolio optimization using custom defined function and minimize libary"""

#Define the lambda parameter
lambd = 0.6

#Define the objective function
def objective_function(x, lambd, Q, a):
    return -np.dot(a,x) + lambd*np.dot(x,np.dot(Q,x))

#Define the optimization constraints
cons = {'type':'eq', 'fun':lambda x: np.sum(x)-1}

minimized_weights = []

for i in range(n):
  current_p = portfolios[i]
  current_cm = cov_matrix_annual[i]
  current_w = np.array(list(initial_weights[i].values()))
  current_r = returns[i]
  #Run the optimization
  current_min = minimize(objective_function, current_w, args=(lambd,current_cm,np.mean(current_r, axis=0)), constraints=cons, bounds=[(0,1) for j in range(len(current_p))])
  minimized_weights.append(current_min)

#Create array of dictionaries to contain the stocks of each portfolio and the minimized weights
optimized_weights =[]

for stocks, peso_portfolio in zip(portfolios, minimized_weights):
    peso_dict = {}
    for stock, peso in zip(stocks, peso_portfolio.x):
        peso_dict[stock] = peso
    optimized_weights.append(peso_dict)

# Capitale iniziale
capital = 10000

# Array vuoto per i dizionari delle quantità
portfolio_quantities_opt3 = []

# Loop sui portfoli per creare i dizionari delle quantità
for i, portfolio in enumerate(portfolios):
    # Accedi ai pesi normalizzati per questo portfolio
    weights_dict = optimized_weights[i]
    # Calcola le quantità di denaro da allocare per ogni stock
    amounts = [weights_dict[stock] * capital for stock in portfolio]
    # Crea un dizionario delle quantità per questo portfolio
    portfolio_dict = dict(zip(portfolio, amounts))
    # Aggiungi il dizionario alla lista di dizionari delle quantità
    portfolio_quantities_opt3.append(portfolio_dict)

# Array vuoto per i nuovi dizionari contenenti le azioni e il numero di azioni acquistabili
portfolio_stocks_opt3 = []

# Loop sui portfoli per creare i nuovi dizionari
for i, portfolio_dict in enumerate(portfolio_quantities_opt3):
    # Crea un nuovo dizionario per questo portfolio
    new_dict = {}
    for stock, investment in portfolio_dict.items():
        # Accedi al prezzo di chiusura per questa azione
        closing_price = close_prices_t0[i][stock]
        # Calcola la quantità di azioni acquistabile in base all'investimento e il prezzo di chiusura
        quantity = (investment / closing_price)
        # Aggiungi l'azione e il numero di azioni acquistabili al dizionario
        new_dict[stock] = quantity
    # Aggiungi il nuovo dizionario alla lista di nuovi dizionari
    portfolio_stocks_opt3.append(new_dict)

# Array vuoto per i nuovi dizionari contenenti le azioni e il numero di azioni acquistabili
portfolio_quantities_tf_opt3 = []

# Loop sui portfoli per creare i nuovi portfolio_stocks_opt1
for i, portfolio_dict in enumerate(portfolio_stocks_opt3):
    # Crea un nuovo dizionario per questo portfolio
    new_dict = {}
    for stock, investment in portfolio_dict.items():
        # Accedi al prezzo di chiusura per questa azione
        closing_price = close_prices_tf[i][stock]
        # Calcola la quantità di azioni acquistabile in base all'investimento e il prezzo di chiusura
        quantity = (investment * closing_price)
        # Aggiungi l'azione e il numero di azioni acquistabili al dizionario
        new_dict[stock] = quantity
    # Aggiungi il nuovo dizionario alla lista di nuovi dizionari
    portfolio_quantities_tf_opt3.append(new_dict)

#PERFOMANCES AFTER OPTIMIZATION 3: CUSTOM FUNCTION
opt_returns = []
opt_annual_return = []
opt_risk = []
opt_percentages = []

for i in range(n):
  current_min_w = minimized_weights[i]
  current_r = returns[i]
  current_cm = cov_matrix_annual[i]

  #Calculate the expected return
  expected_return = np.dot(current_min_w.x,np.mean(current_r, axis=0))*100
  opt_returns.append(expected_return)

  #Calculate the annual return
  annual_return = expected_return*252
  opt_annual_return.append(annual_return)

  #Calculate the expected risk
  expected_risk = np.sqrt(np.dot(current_min_w.x,np.dot(current_cm,current_min_w.x)))*100
  opt_risk.append(expected_risk)

  #calculate percentage for each stock
  percentages = current_min_w.x/np.sum(current_min_w.x)
  opt_percentages.append(percentages)

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Plot the optimized portfolios
plt.scatter(opt_risk, opt_returns, alpha=0.3, label='Optimized Portfolios')

# Interpolate the efficient frontier
f = interp1d(opt_risk, opt_returns, kind='cubic')
x = np.linspace(min(opt_risk), max(opt_risk), num=100)
y = f(x)

# Plot the efficient frontier
plt.plot(x, y, 'r-', label='Efficient Frontier')

plt.xlabel('Expected Risk')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.legend()
plt.show()

"""#Risultati dei portfoli in termini di investimento monetario, capitale iniziale di 10.000 euro"""

#NO OPTIMIZATION
print("PERFOMANCES WITHOUT ANY OPTIMIZATION\n")
# Creazione di una lista di dizionari con i dati di investimento e ritorno per ogni portfolio
portfolio_data = []
for i in range(n):
    data = {"Portfolio": i+1,
            "Investimento iniziale": sum(portfolio_quantities[i].values()),
            "Ritorno finale": sum(portfolio_quantities_tf[i].values())}
    portfolio_data.append(data)

# Creazione di un dataframe pandas con la lista di dizionari
df = pd.DataFrame(portfolio_data)

# Visualizzazione della tabella
print(df.to_string(index=False))

print(portfolio_quantities_tf[0].values())

import matplotlib.pyplot as plt
i=9
  # Definizione dei dati da visualizzare
anni = ['Gen 2017', 'Gen 2019', 'Gen 2022']
budget_2017 = 0
budget_2019 = 10000
budget_2022 = [sum(portfolio_quantities_tf[i].values()), sum(portfolio_quantities_tf_opt1[i].values()),
               sum(portfolio_quantities_tf_opt2[i].values()),sum(portfolio_quantities_tf_opt3[i].values())]
colors = ['blue', 'green', 'red', 'purple']
color_2019 = 'black'
color_2017 = 'yellow'

plt.title(f"Andamento del portafoglio {i+1}")
# Creazione del grafico
plt.scatter([anni[0]]*1, [budget_2017], marker='o', s=100, color=(color_2017))
plt.scatter([anni[1]]*1, [budget_2019], marker='o', s=100, color=color_2019)
for i in range(len(budget_2022)):
  plt.scatter([anni[2]]*1, [budget_2022[i]], marker='o', s=100, color=colors[i])
  plt.xticks(anni)
  plt.ylim(bottom=2000)
  plt.xlabel('Anno')
  plt.ylabel('Budget (€)')
  plt.legend(['2017','Inital Budget', 'No opt', 'Opt 1', 'Opt 2','Opt 3'], loc='upper left')
#Download plot
plt.savefig("fig2.pdf")
from google.colab import files
files.download("fig2.pdf")
plt.show()

#PERFOMANCES AFTER OPTIMIZATION 1: MAXIMIZE SHARPE RATIO
print("PERFOMANCES AFTER OPTIMIZATION 1: MAXIMIZE SHARPE RATIO\n")
# Creazione di una lista di dizionari con i dati di investimento e ritorno per ogni portfolio
portfolio_data_opt1 = []
for i in range(n):
    data = {"Portfolio": i+1,
            "Investimento iniziale": sum(portfolio_quantities_opt1[i].values()),
            "Ritorno finale": sum(portfolio_quantities_tf_opt1[i].values())}
    portfolio_data_opt1.append(data)

# Creazione di un dataframe pandas con la lista di dizionari
df_opt1 = pd.DataFrame(portfolio_data_opt1)

# Visualizzazione della tabella
print(df_opt1.to_string(index=False))

#PERFOMANCES AFTER OPTIMIZATION 2: MINIMIZE VOLATILITY
print("PERFOMANCES AFTER OPTIMIZATION 2: MINIMIZE VOLATILITY\n")
# Creazione di una lista di dizionari con i dati di investimento e ritorno per ogni portfolio
portfolio_data_opt2 = []
for i in range(n):
    data = {"Portfolio": i+1,
            "Investimento iniziale": sum(portfolio_quantities_opt2[i].values()),
            "Ritorno finale": sum(portfolio_quantities_tf_opt2[i].values())}
    portfolio_data_opt2.append(data)

# Creazione di un dataframe pandas con la lista di dizionari
df_opt2 = pd.DataFrame(portfolio_data_opt2)

# Visualizzazione della tabella
print(df_opt2.to_string(index=False))

#PERFOMANCES AFTER OPTIMIZATION 3: CUSTOM FUNCTION
print("PERFOMANCES AFTER OPTIMIZATION 3: CUSTOM FUNCTION\n")
# Creazione di una lista di dizionari con i dati di investimento e ritorno per ogni portfolio
portfolio_data_opt3 = []
for i in range(n):
    data = {"Portfolio": i+1,
            "Investimento iniziale": sum(portfolio_quantities_opt3[i].values()),
            "Ritorno finale": sum(portfolio_quantities_tf_opt3[i].values())}
    portfolio_data_opt3.append(data)

# Creazione di un dataframe pandas con la lista di dizionari
df_opt3 = pd.DataFrame(portfolio_data_opt3)

# Visualizzazione della tabella
print(df_opt3.to_string(index=False))

"""#risultati in termini di rendimento e volatilità"""

#RESULTS WITHOUT OPTIMIZATION
# Creazione di una lista di dizionari con i dati di perfomance per ogni portfolio
perf_data = []
for j in range(len(percent_variances)):
    data = {"Portfolio": f"{j+1}",
            "Exp. ann. return": f"{percent_returns[j]:.2f}%",
            "Ann. volatility (risk)": f"{percent_volatilities[j]:.2f}%",
            "Ann. variance": f"{percent_variances[j]:.2f}%"}
    perf_data.append(data)

# Creazione di un dataframe pandas con la lista di dizionari
df = pd.DataFrame(perf_data)

# Visualizzazione della tabella
print(df.to_string(index=False))

# Salvataggio del dataframe in un file CSV
df.to_csv('performance.csv', index=False)

#PERFOMANCES AFTER OPTIMIZATION 1: MAXIMIZE SHARPE RATIO
#Print the results
print("PERFOMANCES AFTER OPTIMIZATION 1: MAXIMIZE SHARPE RATIO\n")
for i in range(len(frontiers_sharpe)):
  print(f"Portfolio '{i}':")
  frontiers_sharpe[i].portfolio_performance(verbose=True)
  print("\n")

#PERFOMANCES AFTER OPTIMIZATION 2: MINIMIZE VOLATILITY
#Print the results
print("PERFOMANCES AFTER OPTIMIZATION 2: MINIMIZE VOLATILITY\n")
for i in range(len(frontiers_vol)):
  print(f"Portfolio '{i}':")
  frontiers_vol[i].portfolio_performance(verbose=True)
  print("\n")

#PERFOMANCES AFTER OPTIMIZATION 3: CUSTOM FUNCTION
# Creazione di una lista di dizionari con i dati di perfomance per ogni portfolio
perf_data_opt3 = []
for j in range(n):
    data = {"Portfolio": f"{j+1}",
            "Exp. ann. return": f"{opt_annual_return[j]:.2f}%",
            "Ann. volatility (risk)": f"{opt_risk[j]:.2f}%"}
    perf_data_opt3.append(data)

# Creazione di un dataframe pandas con la lista di dizionari
df_opt3 = pd.DataFrame(perf_data_opt3)

# Visualizzazione della tabella
print("PERFOMANCES AFTER OPTIMIZATION 3: CUSTOM FUNCTION\n")
print(df_opt3.to_string(index=False))