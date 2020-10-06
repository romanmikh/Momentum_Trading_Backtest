from scipy.stats import linregress
from datetime import datetime
import matplotlib.pyplot as plt
import backtrader as bt
import pandas as pd
import numpy as np

# Implementation and backtesting of the momentum trading strategy, following procedure outlined in:
# https://www.pdfdrive.com/stocks-on-the-move-beating-the-market-with-hedge-fund-momentum-strategies-d194677183.html

# Data Acquisition and Processing (not part of this code):
# - obtained list of companies currently in the S&P500 from https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
# - obtained historical closing prices for 3000 US publicly-traded companies from Quandl since the 1990s:
#    https://www.quandl.com/databases/WIKIP/usage/export (warning: >450MB file)
# - reduced Quandl data to stocks in the S&P500 only
# - 'SP500_Stock_Data_Sample' should be downloaded from: https://github.com/romanmikh/Momentum_Trading_Backtest
#   'SP500_Stock_Data_Sample' contains only S&P500 companies beginning with 'A', to minimise download time + runtime


"""
Key components of the strategy, as described in Andreas F. Clenow. “Stocks on the Move: Beating the Market with Hedge 
Fund Momentum Strategies" can be summarised as follows:

1) Momentum is defined as the product of the annualised exponential regression slope of the previous 90 days of closing
   prices and the R^2 regression coefficient.

2) For each stock: Position size = AccountValue*RiskFactor/AverageTrueRange_20. 
   - AccountValue = value of the entire trading account
   - RiskFactor = arbitrary number that sets a target daily impact for the stock. (10bp used in these calculations)
   - AverageTrueRange_20 = (price_max - price_min)/20, where 20 days has been chosen arbitrarily by the author
 
3) Analogous to the moving average crossover strategy, new positions will only be opened if the S&P500 is above
   its 200-day moving average.

4) Positions are revised weekly according to the ranks of momenta. Any stocks outside the top 20% are sold and remaining
   cash is used to buy more stocks in the top 20%, according to the position size weightings from 2). 
   
5) Updated Average True Range values are applied fortnightly (every 10 trading days), though trades only happen weekly. 

In the first part of the code (lines 44 - 92) we load S&P500 data, define the momentum of a stock and rank the stocks
within our dataset according to momenta. This is achieved using exponential regression and rolling the 90-day average:
"""

# load the directory 'SP500_Stock_Data_Sample', available at https://github.com/romanmikh/Momentum_Trading_Backtest
SP500_acronyms = pd.read_csv('SP500_Stock_Data_Sample/SP500_acronyms.csv', header=None)[1].tolist()
stocks_data = ((pd.concat([pd.read_csv(f"SP500_Stock_Data_Sample/{company}.csv", index_col='date', parse_dates=True)
                           ['close'].rename(company) for company in SP500_acronyms], axis=1, sort=True)))

# define a function to quantify momentum, to then apply 90-day rolling calculation
def momentum_func(closing_prices):
    """
    :param closing_prices:
    :return: annualised (trading year) regression slope * regression coefficient squared
    """
    log_closing_prices = np.log(closing_prices)  # exp. regression found by first rescaling prices by natural log
    time = np.arange(len(log_closing_prices))
    slope, _null_, rvalue, _null_, _null_ = linregress(time, log_closing_prices)
    return ((1 + slope) ** 252) * (rvalue ** 2)


# calculate 90-day roll of all stocks in stocks_data using momentum_func
momenta = stocks_data.copy(deep=True)  # deep=True copies indices
for company in SP500_acronyms:
    momenta[company] = stocks_data[company].rolling(90).apply(momentum_func, raw=False)
    # raw=False passes each row or column as a Series to the function

# rank stocks according to their momenta and visually compare the highest 3 performers with regression plots
n = 3
top_performers = momenta.max().sort_values(ascending=False).index[:n]
print(f'The {n} top momentum performing stocks are {str(top_performers)[7:27]}.')

plt.figure(figsize=(12, 8))
plt.xlabel('Time/days')
plt.ylabel('Stock Price/USD')

for stock in top_performers:
    max_mom_index = momenta[stock].idxmax()
    end = momenta[stock].index.get_loc(max_mom_index)
    log_returns = np.log(stocks_data[stock].iloc[end - 90: end])
    time = np.arange(len(log_returns))
    slope, intercept, r_value, p_value, std_err = linregress(time, log_returns)
    plt.plot(np.arange(180), stocks_data[stock][end - 90:end + 90], label=f"{stock}")
    if stock == top_performers[2]:
        plt.plot(time, np.e ** (intercept + slope * time), color='red', label="Regression curves")
    else:
        plt.plot(time, np.e ** (intercept + slope * time), color='red')  # rescale from natural log

print('Please close the plot to continue running the code.')
plt.legend()
plt.grid()
plt.show()


# We observe a close match between the regression plots and their corresponding stocks. Outside of the 90 day range of
# the regression plots, the stocks do not continue on the path the regression curve would suggest. This is because our
# objective was only to order the stocks by their momenta, and not to forecast future behaviour. We have run the code
# using a dataset of only 51 stocks (those whose acronyms begin with 'A' to save time, though the algorithm can handle
# bigger datasets, and is more insightful when ran for the entire S&P500 portfolio).


# In this second part of the code we implement the momentum indicator and our strategy, and backtest to find the Sharpe
# ratio, normalised annual return and maximum drawdown of the strategy. We begin by defining momentum (as before) and
# our strategy as classes, according to the 5 axioms stated in the introduction.
#
# For backtesting I have chosen the backtrader library: https://algotrading101.com/learn/backtrader-for-backtesting/
# Backtrader iterates through historical data to evaluate our strategy in the market + simulates the execution of trades
# Backtester allows S&P500 data to be imported directly from the Yahoo Finance API (up until ~20.02.2018) which is then
# used to evaluate our strategy.


# implementing the momentum indicator and our strategy - based on source code from https://teddykoker.com/

class Momentum_indicator(bt.Indicator):  # method in a class?
    lines = ('trend',)
    params = (('period', 90),)  # rolling defined in params by 90 min period

    def __init__(self):  # constructor, no arguments? Should take attributes to be passed into class later? But indic?
        self.addminperiod(self.params.period)

    def regression(self):
        """
        :return: annualised (trading year) regression slope * regression coefficient squared
        """
        returns = np.log(self.data.get(size=self.p.period))
        time = np.arange(len(returns))
        slope, _null_, rvalue, _null_, _null_ = linregress(time, returns)
        annualized = (1 + slope) ** 252
        self.lines.trend[0] = annualized * (rvalue ** 2)  # same as momentum function in part 1

class Momentum_strategy(bt.Strategy):
    def __init__(self):  # constructor
        """
        Find 200-day moving average
        """
        self.counter = 0
        self.indicators = dict()
        self.SP500_Stock_Data_Sample = self.datas[0]
        self.stocks_data = self.datas[1:]  # ignore header
        self.SP500_Stock_Data_Sample_sma200 = bt.indicators.SimpleMovingAverage(self.SP500_Stock_Data_Sample.close,
                                                                                period=200)
        for stock in self.stocks_data:
            self.indicators[stock] = dict()  # make a dictionary of indicators, key = ind, value comes from Mom class and backtrader
            self.indicators[stock]["momentum_func"] = Momentum_indicator(stock.close, period=90)  # close stock after 90 days?
            self.indicators[stock]["simple_moving_average_100"] = bt.indicators.SimpleMovingAverage(stock.close, period=100)
            self.indicators[stock]["average_true_range_20"] = bt.indicators.ATR(stock, period=20)

    def prenext(self):
        """
        Call next() even when data is not available for all SP500_acronyms
        """
        self.next()  # keeps wheels rolling

    def rebalance_portfolio(self):
        """
        Ensure we only look at data that we can have indicators for, sell when outside top 20% or stock below its
        simple_moving_average_100. Buy stocks with remaining cash.
        """
        self.rankings = list(filter(lambda stock: len(stock) > 100, self.stocks_data))  # remove stocks with < 100 days
        self.rankings.sort(key=lambda stock: self.indicators[stock]["momentum_func"][0])  # sort by momenta
        number_of_stocks = len(self.rankings)

        # sell stocks outside the top 20% of momenta or stock underperforming
        for rank, val in enumerate(self.rankings):
            if self.getposition(self.data).size:  # sell/close(val) if outside top 20% or if below 100-day avg
                if rank > number_of_stocks * 0.2 or val < self.indicators[val]["simple_moving_average_100"]:
                    self.close(val)  # closes opened file
        if self.SP500_Stock_Data_Sample < self.SP500_Stock_Data_Sample_sma200:
            return  # do nothing if total portfolio is below its average (?)

        # buy stocks with remaining cash
        for rank, val in enumerate(self.rankings[:int(number_of_stocks * 0.2)]):  # enumerate for top 20%
            cash = self.broker.get_cash()  # cash available, broker(bt) method
            value = self.broker.get_value()  # not specified where from?
            if cash <= 0:
                break
            if not self.getposition(self.data).size:  # determining position size based on formula
                size = value * 0.001 / self.indicators[val]["average_true_range_20"] # risk_factor = 10bp
                self.buy(val, size=size)

    def rebalance_positions(self):
        number_of_stocks = len(self.rankings)
        if self.SP500_Stock_Data_Sample < self.SP500_Stock_Data_Sample_sma200:
            return  # check

        # rebalance all stocks_data
        for rank, val in enumerate(self.rankings[:int(number_of_stocks * 0.2)]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break
            # using equation 2), we essentially weigh each stock according to its risk
            size = value * 0.001 / self.indicators[val]["average_true_range_20"]  # risk_factor = 10bp
            self.order_target_size(val, size)  # give target size,

    def next(self):
        """
        Set weekly trading and fortnightly updates of average_true_range_20
        """
        if self.counter % 5 == 0:
            self.rebalance_portfolio()  # rebalance portfolio weekly (trading)
        if self.counter % 10 == 0:
            self.rebalance_positions()  # rebalance positions fortnightly (if momentum is outside the top 20%)
        self.counter += 1


print('Momentum indicator and strategy initialised.')

# now we run the backtest using the cerebro engine: https://algotrading101.com/learn/backtrader-for-backtesting/
cerebro = bt.Cerebro(stdstats=False)
cerebro.broker.set_coc(True)
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0) # set risk-free rate to 0%
cerebro.addstrategy(Momentum_strategy)

# add S&P500 data from Yahoo Finance API using the backtrader library directly. Though it has been discontinued, we
# can use historical data up till roughly 2018. 2013.07.1 is roughly the earliest 'SP500_Stock_Data_Sample' goes back.
SP500_Stock_Data_Sample = bt.feeds.YahooFinanceData(dataname='SPY', fromdate=datetime(2013, 7, 1),
                                                    todate=datetime(2018, 1, 1))
cerebro.adddata(SP500_Stock_Data_Sample)

for company in SP500_acronyms:
    df = pd.read_csv(f"SP500_Stock_Data_Sample/{company}.csv", parse_dates=True, index_col=0)
    if len(df) > 100:  # data must be long enough to compute 100 day SMA
        cerebro.adddata(bt.feeds.PandasData(dataname=df, plot=False))
    else:
        continue
# why our data?


# finally, now that we have added all data and strategies to the cerebro engine:
print('Running cerebro engine operations...')
results = cerebro.run()

print('Cerebro engine operations performed successfully.')
print(f"Sharpe Ratio achieved: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
print(f"Normalised Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
print(f"Maximum Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

cerebro.plot(iplot=False, figsize=(12, 8))
print('Please close the plot to complete the execution of the code.')
plt.show()

# These results show that the algorithm yields a return of 10.49% on average with a maximum drawdown of 16.23% and
# Sharpe ratio of 1.666. This underperformed the S&P500 over the same time period (compound annual growth rate of
# roughly 12.70% for the S&P500 between 2013 and 2018), and with lower volatility (S&P500 maximum drawdown roughly
# 13.5% and Sharpe ratio 1.07).
#
# With the full S&P500 dataset, we would expect a normalised annual return of 16.51%, a maximum drawdown of 12.93%
# and a Sharpe ratio of 1.426.
#
# While the algorithm is basic relative to modern competitive code, it can be made better by optimising various para-
# meters, applying filters and leveraging the portfolio.
