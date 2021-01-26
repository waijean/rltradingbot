import numpy as np

import itertools


class MultiStockEnv:
    """
    A 3-stock trading environment.
    State: vector of size 7 (n_stock * 2 + 1)
      - # shares of stock 1 owned
      - # shares of stock 2 owned
      - # shares of stock 3 owned
      - price of stock 1 (using daily close price)
      - price of stock 2
      - price of stock 3
      - cash owned (can be used to purchase more stocks)
    Action: categorical variable with 27 (3^3) possibilities
      - for each stock, you can:
      - 0 = sell all shares
      - 1 = sell 1/2 shares
      - 2 = hold
      - 3 = buy all shares
      - 4 = buy 1/2 shares
    """

    def __init__(self, data, initial_investment=20000):
        # data
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape

        # instance attributes
        self.actions_dict = {
            'sell all': 0,
            'sell half': 1,
            'hold': 2,
            'buy all': 3,
            'buy half': 4
        }
        self.n_actions = len(self.actions_dict)
        self.percentage_fees = 0.00
        self.fixed_fees = 0
        self.initial_investment = initial_investment
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        self.action_space = np.arange(self.n_actions ** self.n_stock)

        # action permutations
        # returns a nested list with elements like:
        # [0,0,0]
        # [0,0,1]
        # [0,0,2]
        # [0,1,0]
        # [0,1,1]
        # etc.
        # 0 = sell
        # 1 = hold
        # 2 = buy
        self.action_list = list(
            map(list, itertools.product(range(self.n_actions), repeat=self.n_stock))
        )

        self.action_counts = {}
        # calculate size of state
        self.state_dim = self.n_stock * 2 + 1

        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        self.action_counts = np.zeros(self.n_actions)
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space

        # additional data
        info = {}

        # get current value before performing the action
        prev_val = self._get_val()

        # update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        cur_val = self._get_val()

        # reward is the increase in porfolio value
        reward = cur_val - prev_val

        # done if we have run out of data
        done = self.cur_step == self.n_step - 1

        # store the current value of the portfolio here
        info['cur_val'] = cur_val
        # store action counts
        info['action_counts'] = self.action_counts

        # conform to the Gym API
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[: self.n_stock] = self.stock_owned
        obs[self.n_stock : 2 * self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self):
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _get_fees(self, stock, shares=1):
        total_price = self.stock_price[stock] * shares
        percentage_fees_value = total_price * self.percentage_fees
        if percentage_fees_value >= self.fixed_fees:
            return percentage_fees_value
        else:
            return self.fixed_fees

    def _trade(self, action):
        # index the action we want to perform
        # - 0 = sell all shares
        # - 1 = sell 1/2 shares
        # - 2 = hold
        # - 3 = buy all shares
        # - 4 = buy 1/2 shares
        # e.g. [2,1,0] means:
        # buy first stock
        # hold second stock
        # sell third stock
        action_vec = self.action_list[action]

        # determine which stocks to buy or sell

        action_idx = [ [] for _ in range(self.n_actions) ]
        # [[1],[0],[2],[],[]]

        for i, action_code in enumerate(action_vec):
            action_idx[action_code].append(i)

        def get_stocks(action_name):
            return action_idx[self.actions_dict[action_name]]

        # Count transactions
        action_counts = {}
        for i in range(self.n_actions):
            self.action_counts[i] += sum(action_idx[i])

        # sell any stocks we want to sell
        # then buy any stocks we want to buy
        for stock in get_stocks('sell all'):
            if self.stock_owned[stock] > 0:
                total_selling_price = self.stock_price[stock] * self.stock_owned[stock]
                total_fees = self._get_fees(stock, self.stock_owned[stock])
                assert self.cash_in_hand + (total_selling_price - total_fees) >0
                self.cash_in_hand += (total_selling_price - total_fees)
                self.stock_owned[stock] = 0
        for stock in get_stocks('sell half'):
            if self.stock_owned[stock] > 0:
                half_shares = (self.stock_owned[stock] // 2) + 1
                total_selling_price = self.stock_price[stock] * half_shares
                total_fees = self._get_fees(stock, half_shares)
                assert self.cash_in_hand + (total_selling_price - total_fees) >0
                self.cash_in_hand += (total_selling_price - total_fees)
                self.stock_owned[stock] = self.stock_owned[stock] - half_shares

        all_shares_price = self.stock_price[get_stocks('buy all')].sum()
        half_shares_price = self.stock_price[get_stocks('buy half')].sum()
        # If there's a mix of buying all or half the shares of different
        # stocks buy shares at twice the rate for stocks that you plan to
        # buy as much shares as posible actions
        if get_stocks('buy all') and get_stocks('buy half'):
            total_shares_price = 2*all_shares_price + half_shares_price
            def get_total_fees(n_batches):
                buy_all_fees = sum(self._get_fees(stock, 2*n_batches)
                                for stock in get_stocks('buy all'))
                buy_half_fees = sum(self._get_fees(stock, 1*n_batches)
                                 for stock in get_stocks('buy half'))
                return buy_all_fees + buy_half_fees
            n_batches = 1
            total_fees = get_total_fees(n_batches)
            while self.cash_in_hand >= total_shares_price*n_batches + total_fees:
                for stock in get_stocks('buy half'):
                    self.stock_owned[stock] += 1  # buy one share
                for stock in get_stocks('buy all'):
                    self.stock_owned[stock] += 2  # buy two shares
                n_batches += 1
                total_fees = get_total_fees(n_batches)
            if n_batches > 1:
                final_fees = get_total_fees(n_batches - 1)
                assert self.cash_in_hand > total_shares_price*(n_batches-1) + final_fees
                self.cash_in_hand -= (total_shares_price*(n_batches-1) + final_fees)
        # Buy all shares if there are no "buy half" actions
        elif get_stocks('buy all'):
            total_shares_price = all_shares_price
            def get_total_fees(n_batches):
                return sum(self._get_fees(stock, n_batches) for stock in get_stocks('buy all'))
            n_batches = 1
            total_fees = get_total_fees(n_batches)
            while self.cash_in_hand >= total_shares_price*n_batches + total_fees:
                for stock in get_stocks('buy all'):
                        self.stock_owned[stock] += 1  # buy one share
                n_batches += 1
                total_fees = get_total_fees(n_batches)
            if n_batches > 1:
                final_fees = get_total_fees(n_batches - 1)
                assert self.cash_in_hand > total_shares_price*(n_batches-1) + final_fees
                self.cash_in_hand -= (total_shares_price*(n_batches-1) + final_fees)
        # When buying only half shares buy half of the shares you would be able
        # to buy
        elif get_stocks('buy half'):
            total_shares_price = half_shares_price
            def get_total_fees(n_batches):
                return sum(self._get_fees(stock, n_batches) for stock in get_stocks('buy half'))
            n_batches = 1
            total_fees = get_total_fees(n_batches)
            while self.cash_in_hand/2 >= total_shares_price*n_batches + total_fees:
                for stock in get_stocks('buy half'):
                    self.stock_owned[stock] += 1  # buy one share
                n_batches += 1
                total_fees = get_total_fees(n_batches)
            if n_batches > 1:
                final_fees = get_total_fees(n_batches - 1)
                assert self.cash_in_hand > total_shares_price*(n_batches-1) + final_fees
                self.cash_in_hand -= (total_shares_price*(n_batches-1) + final_fees)
