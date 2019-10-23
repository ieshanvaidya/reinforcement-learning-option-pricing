import pytest
from optionpricing import *

"""
Delta neutral policy example 1 (values simulated in Excel)
"""

m = -100000
K = 50
T = 0.3846
dt = 0.01923
r = 0.05
sigma = 0.2
mu = 0.13
cash = 300000

stock_prices = [49.00, 48.12, 47.37, 50.25, 51.75, 53.12, 53.00, 51.87, 51.38, 53.00, 49.88, 48.50, 49.88, 50.37, 52.13, 51.88, 52.87, 54.87, 54.62, 55.87, 57.25]
call_prices = [2.4005, 1.8875, 1.4872, 2.8366, 3.7112, 4.6235, 4.4374, 3.5037, 3.0629, 4.1451, 1.9235, 1.1481, 1.6916, 1.8435, 2.9124, 2.5911, 3.2668, 5.0356, 4.7235, 5.9181, 7.2500]
wealths = [59945.8848, 62860.5771, 66402.3575, 44766.9317, 43885.7361, 44246.6525, 49822.3442, 52288.7295, 58360.7430, 56089.6020, 29020.5334, 27911.0932, 28470.0620, 37175.2607, 31290.5146, 40389.2484, 44210.2960, 35989.5192, 37817.6429, 37076.4586, 36849.9315]
pnls = [wealths[i] - wealths[i - 1] for i in range(1, len(wealths))]
deltas = [0.5216, 0.4580, 0.4000, 0.5963, 0.6930, 0.7738, 0.7713, 0.7062, 0.6742, 0.7865, 0.5502, 0.4128, 0.5425, 0.5905, 0.7682, 0.7592, 0.8650, 0.9783, 0.9899, 1.0000, 1.0000]
gammas = [0.0655, 0.0682, 0.0693, 0.0674, 0.0612, 0.0527, 0.0550, 0.0664, 0.0730, 0.0597, 0.0905, 0.0965, 0.1014, 0.1051, 0.0861, 0.0968, 0.0740, 0.0197, 0.0126, 0.0001, 0.0000]
costs = [2555848.0065, 2252280.7596, 1979776.8375, 2967882.3788, 3471013.9954, 3903933.1975, 3894270.6523, 3560139.5403, 3399340.6745, 3998004.4046, 2822876.4560, 2159117.9642, 2808212.8565, 3052973.6839, 3982197.6208, 3939233.3603, 4502678.4955, 5128159.0439, 5196611.0301, 5258005.7243, 5263207.3185]
nstocks = [52160, 45800, 40002, 59628, 69295, 77382, 77129, 70615, 67419, 78653, 55017, 41275, 54247, 59053, 76822, 75920, 86506, 97826, 98989, 99998, 100001]

portfolio =namedtuple('Portfolio', ['S', 'call', 'n', 'm', 'L', 'cash'])

def test_call():
    for i, stock_price in enumerate(stock_prices):
        call = compute_call(stock_price, K, T - i * dt, r, sigma)
        assert pytest.approx(call, abs = 0.0001) == call_prices[i]

def test_greeks():
    for i, stock_price in enumerate(stock_prices):
        delta, gamma = compute_greeks(stock_price, K, T - i * dt, r, sigma)
        assert pytest.approx(delta, abs = 0.0001) == deltas[i]
        assert pytest.approx(gamma, abs = 0.0001) == gammas[i]

def test_pnl():
    portfolios = []
    for i, stock_price in enumerate(stock_prices):
        call = compute_call(stock_price, K, T - i * dt, r, sigma)
        cash_pos = cash - costs[i]
        #portfolio = {'S': stock_price, 'call': call, 'cash': cash_pos, 'n': nstocks[i], 'm': m, 'L': 1}
        portfolios.append(portfolio(stock_price, call, nstocks[i], m, 1, cash_pos))

    for i in range(1, len(portfolios)):
        pnl = compute_pnl(portfolios[i - 1], portfolios[i])
        assert pytest.approx(pnl, abs = 0.0001) == pnls[i - 1]
