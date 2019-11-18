import gym
from gym import error, spaces, utils
import numpy as np
import random
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

def compute_pnl(init_portfolio, final_portfolio):
    init_wealth = compute_wealth(init_portfolio)
    final_wealth = compute_wealth(final_portfolio)

    return final_wealth - init_wealth


def compute_wealth(portfolio):
    option_value = portfolio.call * portfolio.m * portfolio.L
    stock_value = portfolio.S * portfolio.n
    cash = portfolio.cash

    return option_value + stock_value + cash


class OptionPricingEnv:
    def __init__(self, config):
        """
        config: Configuration dictionary with k:v as
            S: stock price (float)
            T: days to maturity (int or list of ints)
            L: number of option contracts (int)
            m: number of stocks per option (int)
            n: number of stocks (int)
            K: strike price (float or list of floats)
            D: trading periods per day (int)
            mu: expected rate of return on the stock (float)
            sigma: volatility of stock (float)
            r: risk free rate (float)
            ss: number of steps between trading periods (int)
            kappa: risk aversion (float)
        """
        self.config = config

        self.trading_days = 252
        self.day = 24 / self.trading_days # 24 hours
        self.lots = 1

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

    def configure(self):
        self.S = self.config['S']
        try:
            self.T = random.choice(self.config['T'])
        except TypeError:
            self.T = self.config['T']

        self.L = self.config['L']
        self.m = self.config['m']
        self.n = self.config['n']
        try:
            self.K = random.choice(self.config['K'])
        except TypeError:
            self.K = self.config['K']

        #self.K = K
        self.D = self.config['D']
        self.mu = self.config['mu']
        self.sigma = self.config['sigma'] * np.sqrt(self.trading_days) # Converting sigma/day to sigma/year
        self.r = self.config['r']
        self.ss = self.config['ss']
        self.kappa = self.config['kappa']
        self.multiplier = self.config['multiplier']
        self.ticksize = self.config['ticksize']

        self.S0 = self.S
        self.cash = 0

        #self.init_config = {k: v for k, v in locals().items() if k != 'self'}

        self.t = self.day * self.T
        self.steps = self.T * self.D
        self.dt = self.day / self.D / self.ss

        if not np.isclose(0, (self.t / self.dt) % 1):
            raise ValueError('Mismatch in "time to expiry" and "stochastic time step"')

        h = abs(self.L * self.m)
        l = -h
        num_actions = int((h - l) / self.lots + 1)

        self.high = h

        self.observation_space = spaces.Box(low = np.array([0, 0, 0, -np.inf]), high = np.array([np.inf, np.inf, np.inf, np.inf]))
        self.action_space = spaces.Discrete(num_actions)

        self.action_map = {i: int(l + i * self.lots) for i in range(self.action_space.n)}
        self.inv_action_map = {v: k for k, v in self.action_map.items()}

        self.configured = True
        self.done = False

    def step(self, action, stock_prices = None):
        """
        stock_prices: for deterministic evolution | dtype: list (even if single entry)
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

            states.append([self.S / self.S0, self.t, self.n / self.high, self.K / self.S0])

        self.steps -= 1

        cost = self.multiplier * self.ticksize * (abs(num_stocks) + 0.01 * num_stocks ** 2)
        self.cash -= cost

        pnl = compute_pnl(init_portfolio, self.portfolio)

        reward = (pnl - 0.5 * self.kappa * (pnl ** 2))

        info = {'pnl': pnl, 'dn': num_stocks, 'call': np.array(calls), 'delta': np.array(deltas), 'gamma': np.array(gammas), 'cost': cost}

        self.done = self.steps == 0

        return np.array(states[-1], dtype = np.float32), reward, self.done, info

    def reset(self):
        self.configure()
        return np.array([self.S / self.S0, self.t, self.n / self.high, self.K / self.S0], dtype = np.float32)

    def render(self):
        pass

    def close(self):
        pass
