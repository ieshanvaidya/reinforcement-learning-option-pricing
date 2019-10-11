import gym
from gym import error, spaces, utils
import numpy as np
from scipy.stats import norm
from collections import namedtuple


def compute_call(S, K, t, r, sigma):
    if np.isclose(t, 0):
        return max(0, S - K)

    if t == 0:
        return max(0, S - K)

    d1 = ((np.log(S / K) + (r + sigma ** 2 / 2) * t)) / sigma / np.sqrt(t)
    d2 = d1 - sigma * np.sqrt(t)
    call = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    return call

def compute_greeks(S, K, t, r, sigma):
    if np.isclose(t, 0):
        return 1, 0

    if t == 0:
        return 1, 0

    d1 = ((np.log(S / K) + (r + sigma ** 2 / 2) * t)) / sigma / np.sqrt(t)
    delta = norm.cdf(d1)
    gamma = (1 / np.sqrt(2 * np.pi)) * np.exp(-d1 ** 2 / 2) / (S * sigma * np.sqrt(t))

    return delta, gamma

def compute_pnl(init_portfolio, final_portfolio, transaction_cost):
    init_wealth = compute_wealth(init_portfolio)
    final_wealth = compute_wealth(final_portfolio)

    return final_wealth - init_wealth - transaction_cost


def compute_wealth(portfolio):
    option_value = portfolio.call * portfolio.m * portfolio.L
    stock_value = portfolio.S * portfolio.n
    cash = portfolio.cash

    return option_value + stock_value + cash


class OptionPricingEnv:
    def __init__(self):
        """
        S: stock price
        T: days to maturity
        L: number of option contracts
        m: number of stocks per option
        n: number of stocks
        K: strike price
        D: trading periods per day
        mu: expected rate of return on the stock
        sigma: volatility of stock
        r: risk free rate
        ss: number of steps between trading periods
        kappa: risk aversion
        """
        # Portfolio
        self.S = None
        self.T = None
        self.L = None
        self.m = None
        self.n = None
        self.K = None
        self.D = None
        self.mu = None
        self.sigma = None
        self.r = None
        self.ss = None

        # Optimization
        self.kappa = None

        self.trading_days = 252
        self.day = 24 / self.trading_days # 24 hours
        self.lots = 1
        self.trading_cost = 0

        self.configured = False

    @property
    def call(self):
        return compute_call(self.S, self.K, self.t, self.r, self.sigma)

    @property
    def portfolio(self):
        return namedtuple('Portfolio', ['S', 'call', 'n', 'm', 'L', 'cash'])(self.S, self.call, self.n, self.m, self.L, self.cash)

    @property
    def stock_value(self):
        return self.n * self.S

    @property
    def option_value(self):
        return self.call * self.m * self.L

    @property
    def delta(self):
        delta, gamma = compute_greeks(self.S, self.K, self.t, self.r, self.sigma)
        return delta

    def configure(self, S, T, L, m, n, K, D, mu, sigma, r, ss, kappa, multiplier, ticksize):
        self.S = S
        self.T = T
        self.L = L
        self.m = m
        self.n = n
        self.K = K
        self.D = D
        self.mu = mu
        self.sigma = sigma * np.sqrt(self.trading_days) # Converting sigma/day to sigma/year
        self.r = r
        self.ss = ss
        self.kappa = kappa
        self.multiplier = multiplier
        self.ticksize = ticksize

        self.S0 = S
        self.cash = 0

        self.init_config = {k: v for k, v in locals().items() if k != 'self'}

        self.t = self.day * self.T
        self.steps = self.T * self.D
        self.dt = self.day / self.D / ss

        if not np.isclose(0, (self.t / self.dt) % 1):
            raise ValueError('Mismatch in "time to expiry" and "stochastic time step"')

        h = abs(self.L * self.m)
        l = -h
        num_actions = int((h - l) / self.lots + 1)

        self.high = h

        self.observation_space = spaces.Box(low = np.array([0, 0, -np.inf]), high = np.array([np.inf, np.inf, np.inf]))
        self.action_space = spaces.Discrete(num_actions)

        self.action_map = {i: int(l + i * self.lots) for i in range(self.action_space.n)}
        self.inv_action_map = {v: k for k, v in self.action_map.items()}

        self.configured = True
        self.done = False

    def step(self, action, stock_prices = None):
        """
        stock_prices: for deterministic evolution | list type even if single entry
        """
        if not self.configured:
            raise NotImplementedError('Environment not configured')

        if self.done:
            return

        init_portfolio = self.portfolio

        num_stocks = self.action_map[action]
        self.n = self.n + num_stocks

        states = []
        calls = []
        deltas = []
        gammas = []

        self.cash = self.cash - self.S * num_stocks

        for i, period in enumerate(range(self.ss)):
            if stock_prices is not None:
                self.S = stock_prices[i]

            else:
                ds = self.mu * self.S * self.dt + self.sigma * self.S * np.random.normal() * np.sqrt(self.dt)
                self.S = self.S + ds

            self.t = max(0, self.t - self.dt)

            call = self.call
            delta, gamma = compute_greeks(self.S, self.K, self.t, self.r, self.sigma)

            calls.append(call)
            deltas.append(delta)
            gammas.append(gamma)

            states.append([self.S / self.S0, self.t, self.n / self.high])
            #states.append([self.S, self.t, self.n])

        self.steps -= 1

        cost = self.multiplier * self.ticksize * (abs(num_stocks) + 0.01 * num_stocks ** 2)
        self.cash -= cost

        pnl = compute_pnl(init_portfolio, self.portfolio, self.trading_cost)

        reward = 0.01 * (pnl - 0.5 * self.kappa * (pnl ** 2) - cost)

        info = {'call': np.array(calls), 'delta': np.array(deltas), 'gamma': np.array(gammas)}

        self.done = self.steps == 0

        return np.array(states[-1], dtype = np.float32), reward, self.done, info

    def reset(self):
        # Should return the state
        self.configure(**self.init_config)
        return np.array([self.S / self.S0, self.t, self.n / self.high], dtype = np.float32)
        #return np.array([self.S, self.t, self.n])

    def render(self):
        pass

    def close(self):
        pass

