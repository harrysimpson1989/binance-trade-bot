import random
import sys
from datetime import datetime

import pandas

from binance_trade_bot.auto_trader import AutoTrader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
# matplotlib.use('Agg')
import pandas as pd

def ema(value_now, ema_prev, period, smoothing=2.0):
    """
    Calculate exponential moving average
    """
    ema_now = value_now*(smoothing/(1+period)) + ema_prev*(1-(smoothing/(1+period)))
    return ema_now

class Strategy(AutoTrader):
    def initialize(self):
        super().initialize()
        self.initialize_current_coin()

        # my stuff
        self.history = pd.DataFrame(columns=['time','price'])
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3,1)
        plt.ion()
        plt.show()
        plt.draw()

    def scout(self):
        """
        Scout for potential jumps from the current coin to another coin
        """
        current_coin = self.db.get_current_coin()
        # Display on the console, the current coin+Bridge, so users can see *some* activity and not think the bot has
        # stopped. Not logging though to reduce log size.
        print(
            f"{datetime.now()} - CONSOLE - INFO - I am scouting the best trades. "
            f"Current coin: {current_coin + self.config.BRIDGE} ",
            end="\r",
        )

        current_coin_price = self.manager.get_ticker_price(current_coin + self.config.BRIDGE)

        if current_coin_price is None:
            self.logger.info("Skipping scouting... current coin {} not found".format(current_coin + self.config.BRIDGE))
            return

        # my stuff
        self.history = pandas.concat([
            self.history,
            pd.DataFrame({
                'time': [datetime.now()],
                'price': [current_coin_price],
                'ema_26': [current_coin_price],
                'ema_12': [current_coin_price],
                'macd': [0.0],
                'macd_sma': [0.0],
                'money_gbp':[1000.0],
                'money_btc': [0.0],
            })])
        self.history.reset_index(inplace=True, drop=True)
        if len(self.history) > 100:
            self.history.drop(index=0, inplace=True)

        if len(self.history) < 2:
            return
        else:
            # ema
            ind_last = self.history.index[-1]
            self.history.loc[ind_last, 'ema_26'] = ema(current_coin_price, self.history.iloc[-2]['ema_26'], 26)
            self.history.loc[ind_last, 'ema_12'] = ema(current_coin_price, self.history.iloc[-2]['ema_12'], 12)
            self.history.loc[ind_last, 'macd'] = self.history.loc[ind_last, 'ema_12'] - self.history.loc[ind_last, 'ema_26']
            self.history.loc[ind_last, 'macd_sma'] = ema(self.history.iloc[-1]['macd'], self.history.iloc[-2]['macd_sma'], 9)
            self.history.loc[ind_last, 'sig_diff'] = self.history.loc[ind_last, 'macd'] - self.history.loc[ind_last, 'macd_sma']
            crossed = self.history.iloc[-1]['sig_diff']*self.history.iloc[-2]['sig_diff'] < 0
            slope = self.history.iloc[-1]['sig_diff'] - self.history.iloc[-2]['sig_diff']
            self.history.loc[ind_last, 'money_gbp'] = self.history.iloc[-2]['money_gbp']
            self.history.loc[ind_last, 'money_btc'] = self.history.iloc[-2]['money_btc']

            if crossed:
                if slope < 0 and self.history.loc[ind_last, 'macd'] > -1:
                    # sell condition
                    self.history.loc[ind_last, 'buy_sell'] = -1

                    if self.history.iloc[-2]['money_btc'] > 0:
                        self.history.loc[ind_last, 'money_gbp'] = self.history.iloc[-2]['money_btc'] * current_coin_price
                        self.history.loc[ind_last, 'money_btc'] = 0
                elif slope > 0 and self.history.loc[ind_last, 'macd'] < 1:
                    # buy condition
                    self.history.loc[ind_last, 'buy_sell'] = 1

                    if self.history.iloc[-2]['money_gbp'] > 0:
                        self.history.loc[ind_last, 'money_btc'] = self.history.iloc[-2]['money_gbp'] / current_coin_price
                        self.history.loc[ind_last, 'money_gbp'] = 0
            else:
                self.history.loc[ind_last, 'buy_sell'] = 0

            self.history.loc[ind_last, 'money'] = (self.history.loc[ind_last, 'money_gbp'] + self.history.loc[ind_last, 'money_btc']*current_coin_price)-1000.0

            # plotting
            self.ax1.clear()
            self.history.plot('time', ['price', 'ema_26', 'ema_12'], ax=self.ax1, color=['k', 'b', 'r'])
            self.ax2.clear()
            self.ax2.plot(self.history['time'], [0.0]*len(self.history), color='k', linestyle='--')
            self.history.plot('time', ['macd','macd_sma','buy_sell'], ax=self.ax2, color=['b','y','g'])
            self.ax3.clear()
            self.history.plot('time', 'money', ax=self.ax3, color='g')

            plt.draw()
            plt.pause(0.001)



    def bridge_scout(self):
        current_coin = self.db.get_current_coin()
        if self.manager.get_currency_balance(current_coin.symbol) > self.manager.get_min_notional(
            current_coin.symbol, self.config.BRIDGE.symbol
        ):
            # Only scout if we don't have enough of the current coin
            return
        new_coin = super().bridge_scout()
        if new_coin is not None:
            self.db.set_current_coin(new_coin)

    def initialize_current_coin(self):
        """
        Decide what is the current coin, and set it up in the DB.
        """
        if self.db.get_current_coin() is None:
            current_coin_symbol = self.config.CURRENT_COIN_SYMBOL
            if not current_coin_symbol:
                current_coin_symbol = random.choice(self.config.SUPPORTED_COIN_LIST)

            self.logger.info(f"Setting initial coin to {current_coin_symbol}")

            if current_coin_symbol not in self.config.SUPPORTED_COIN_LIST:
                sys.exit("***\nERROR!\nSince there is no backup file, a proper coin name must be provided at init\n***")
            self.db.set_current_coin(current_coin_symbol)

            # if we don't have a configuration, we selected a coin at random... Buy it so we can start trading.
            if self.config.CURRENT_COIN_SYMBOL == "":
                current_coin = self.db.get_current_coin()
                self.logger.info(f"Purchasing {current_coin} to begin trading")
                self.manager.buy_alt(current_coin, self.config.BRIDGE)
                self.logger.info("Ready to start trading")
