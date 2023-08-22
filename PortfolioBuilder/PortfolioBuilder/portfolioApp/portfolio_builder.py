import math
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime as datetime
from pandas.core.arrays.period import timedelta
import yahoo_fin.stock_info as si
from forex_python.converter import CurrencyRates
import random
from scipy.optimize import minimize
import concurrent.futures

def fetch_live_price(ticker):
  print(ticker)
  try:
    live_price = si.get_live_price(ticker)
  except:
    live_price = math.inf
  return live_price

def getDf():
  df = pd.read_csv('static/etfs_list.csv')
  tickers = df['symbol'].tolist()
  
  adj_close = []
  
  # Use a ThreadPoolExecutor for concurrent processing
  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit tasks for fetching live prices
    futures = [executor.submit(fetch_live_price, ticker) for ticker in tickers]
    
    # Retrieve results as they become available
    for future in concurrent.futures.as_completed(futures):
      adj_close.append(future.result())

  df["adj_close"] = adj_close
  return df
df = getDf()

def getNumberOfEtf(amount):
  num_portfolios = {
    (1, 5000):5,
    (5001, 20000):10,
    (20000, math.inf): 15
  }
  no_etfs = 0
  for k in num_portfolios:
    if amount>=k[0] and amount<=k[1]:
      no_etfs = num_portfolios[k]
  return no_etfs

def getPortfolios(number, etfs):
  if len(etfs) <= number:
    return [etfs]
  portfolios = []
  for i in range(10):
    portfolio = random.sample(etfs, number)
    portfolios.append(portfolio)
  return portfolios

def adj_close(ticker, start, end):
  data = yf.download(ticker, progress=False, start=start, end=end)
  if data.shape[0] == 0:
    data = yf.download(ticker, progress=False)
  return data['Adj Close']

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)
def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

def optimized_portfolio(tickers, log_returns, cov_matrix, risk_free_rate):
  constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
  bounds = [(0, 0.4) for _ in range(len(tickers))]
  initial_weights = np.array([1/len(tickers)]*len(tickers))
  optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
  optimal_weights = optimized_results.x
  result = {}
  result["weights"] = optimal_weights
  result["return"] = expected_return(optimal_weights, log_returns)
  result["volatility"] = standard_deviation(optimal_weights, cov_matrix)
  result["sharpRatio"] = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
  return result

def sharpRatioAndVaR(amount, years, etfs):
  end = datetime.today()
  start = end - timedelta(days=(years*1.5)*365)
  #calculate adj close
  adj_close_df = pd.DataFrame()
  for ticker in etfs:
    adj_close_df[ticker] = adj_close(ticker, start, end)
  adj_close_df.dropna(axis=0, inplace=True)

  #calculate log returns
  log_returns = np.log(adj_close_df/adj_close_df.shift(1))
  log_returns.dropna()

  #calculate covariance matrix
  cov_matrix = log_returns.cov() * 252

  #optimize portfolio
  risk_free_rate = 0.02
  result = optimized_portfolio(etfs, log_returns, cov_matrix, risk_free_rate)
  historical_returns = (log_returns * result['weights']).sum(axis =1)
  days = 5
  range_returns = historical_returns.rolling(window = days).sum()
  # print("before " , range_returns)
  range_returns = range_returns.dropna()
  confidence_interval = 0.99
  # print(historical_returns)
  # print(range_returns)

  VaR = -np.percentile(range_returns, 100 - (confidence_interval * 100))*amount
  result['VaR'] = VaR
  result['etfs'] = etfs
  # print(result)
  return result

def get_expense_ratio(etfs_list):
  expense_ratios = []
  print("etfs_list ", etfs_list)
  for ticker in etfs_list:
    d1 = si.get_analysts_info(ticker)[0]
    ratio = d1.loc[d1[0]=='Expense Ratio (net)', 1].values[0]
    expense_ratios.append(ratio)
  return expense_ratios

def getOptPortfolio(amount, years, risk_level):
  cr = CurrencyRates()
  amount = cr.convert('GBP', 'USD', amount)
  no_etfs = getNumberOfEtf(amount)
  data = df.loc[df["adj_close"]<=amount]
  etfs = list(data['symbol'])
  portfolios = getPortfolios(no_etfs, etfs)
  sharpRatioAndVaRs = []
  for portfolio in portfolios:
    # print("portfolio ",portfolio)
    try:
      sharpRatioAndVaRs.append(sharpRatioAndVaR(amount, years, portfolio))
      # print("sharpRatioAndVaRs ", sharpRatioAndVaRs)
    except:
      pass
  risk_amount = risk_level * amount
  # print("risk_amount ", risk_amount)
  backup_sr = sharpRatioAndVaRs
  sharpRatioAndVaRs = [x for x in sharpRatioAndVaRs if x["VaR"] <= risk_amount]
  # print("sharpRatioAndVaRs ", sharpRatioAndVaRs)
  if len(sharpRatioAndVaRs) == 0:
    backup_sr.sort(key=lambda portfolio:portfolio['VaR'])
    obj = backup_sr[0]
    obj["exp_ratio"] = get_expense_ratio(backup_sr[0]["etfs"])
    return obj
  sharpRatioAndVaRs.sort(key=lambda portfolio:portfolio['sharpRatio'])
  obj = sharpRatioAndVaRs[-1]
  obj["exp_ratio"] = get_expense_ratio(sharpRatioAndVaRs[-1]["etfs"])
  return obj