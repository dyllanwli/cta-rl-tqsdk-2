from datetime import date, datetime
from contextlib import closing


import wandb
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask, BacktestFinished, TqSim, TqAccount
from tqsdk.objs import Quote, Account
from tqsdk.ta import ATR
# from tqsdk.tafunc import ema, get_ticks_info
import pytz

import pandas as pd
import numpy as np


class DynamicLinearRegressionChannels:
    def __init__(self, auth: TqAuth, commission_fee: float = 4.4, volume: int = 1, is_wandb: bool = True):
        self.auth = auth
        self.commission_fee = commission_fee
        self.is_wandb = is_wandb
        self.volume = volume
        self.ema_periods = [5, 10, 20, 60]

    def backtest(
        self,
        symbol: str,
        start_dt=date(2022, 12, 1),
        end_dt=date(2022, 12, 25),
        is_live: bool = False,
    ):
        if is_live:
            acc: Account = TqAccount()  # TBD
            self.api = TqApi(account=acc, auth=self.auth, web_gui=False)
            self.account: Account = self.api.get_account()
        else:
            # use simulation account
            sim = TqSim(init_balance=200000)
            self.api = TqApi(account=sim, auth=self.auth, backtest=TqBacktest(
                start_dt=start_dt, end_dt=end_dt), web_gui=False)
            sim.set_commission(symbol=symbol, commission=self.commission_fee)
            self.account: Account = self.api.get_account()

        print("Subscribing quote")
        quote: Quote = self.api.get_quote(symbol)

        klines_1m = self.api.get_kline_serial(
            symbol, duration_seconds=45, data_length=500)
        self.target_pos_task = TargetPosTask(self.api, symbol, price="ACTIVE")
        if self.is_wandb:
            wandb.init(project="backtest-1", config={"symbol": symbol})
        with closing(self.api):
            try:
                while True:
                    self.api.wait_update()
                    if self.api.is_changing(klines_1m.iloc[-1], "datetime"):

                        # calculate signal time
                        start_time = datetime.now()
                        signal = self.get_signal(klines_1m)
                        end_time = datetime.now()

                        if signal == 1:
                            self.target_pos_task.set_target_volume(self.volume)
                        elif signal == -1:
                            self.target_pos_task.set_target_volume(
                                -self.volume)
                        else:
                            # hold or close position
                            # print("Hold position")
                            pass
                        self.api.wait_update()

                    if self.is_wandb:
                        wandb.log({
                            "singal_time": (end_time - start_time).total_seconds(),
                            "signal": signal,
                            "last_price": quote.last_price,
                            "static_balance": self.account.static_balance,
                            "account_balance": self.account.balance,
                            "commission": self.account.commission,
                        })
            except BacktestFinished:
                print("Backtest done")

    def check_trading_time(self, dt: int):
        """
        close position at the end of the day
        """
        exchange_tz = pytz.timezone('Asia/Shanghai')
        dt = datetime.utcfromtimestamp(dt / 1e9).astimezone(exchange_tz)
        if (dt.hour == 14 and dt.minute > 58) or (dt.hour == 22 or dt.minute > 58):
            return False
        else:
            return True

    def close_position(self):
        self.target_pos_task.set_target_volume(0)

    def get_signal(self, klines_1m: pd.Series, mode: str = "envelope"):
        """
        1: long
        -1: short
        0: close position
        """
        if mode == "base":
            signal = self.nadaraya_watson(
                close=klines_1m['close'], h=5, r=1, x0=25, lag=2)
        elif mode == "envelope":
            signal = self.nadaraya_watson_envelope(
                close=klines_1m['close'], h=6, mult=3, length=500)
        return signal

    def nadaraya_watson_envelope(self, close: pd.Series, h: float = 5, mult: float = 3, length: int = 500):
        """
        Nadaraya-Watson Envelope
            close: close price, 
            h: bandwidth, controls the degree of smoothness of the envelopes , with higher values returning smoother results.
            mult: multiplier, controls the envelope width
            length: determines the number of recent price observations to be used to fit the Nadaraya-Watson Estimator.
        """

        # calculate the Nadaraya-Watson envelope on the last bar
        y = np.zeros(length)

        # calculate the kernel function
        estimation = 0
        for i in range(length):
            current_weight = 0
            cumulative_weight = 0
            for j in range(length):
                w = np.exp(-(np.power(i - j, 2) / (h * h * 2)))
                current_weight += close.iloc[j] * w
                cumulative_weight += w
            y2 = current_weight / cumulative_weight
            estimation += np.abs(y2 - close.iloc[i])
            y[i] = y2

        # calculate the mean absolute error
        mae = estimation / length * mult

        upper_band = y + mae  # upper band
        lower_band = y - mae  # lower band

        if self.crossup(close.values, upper_band):
            signal = -1
        elif self.crossdown(close.values, lower_band):
            signal = 1
        else:
            signal = 0
        return signal

    def nadaraya_watson(self, close: pd.Series, h: float = 3, r: float = 1, x0: int = 25, lag: int = 1):
        """
        Nadaraya-Watson kernel regression
            close: close price, 
            h: bandwidth, number of bars used for the estimation, this is a sliding value that represents the most recent historical bars. Recommended range: 3-50
            r: relative weighting, relative weighting of time frames. As this value approaches zerom the longer time frames will exert more influence on the stimation. As this value approaches
                infinity, behavior of the relational quadratic kernel will become indentical to he gaussian kernel. Recommended range: 0.25-25
            x0: regression start point, bar index on which to start regression. The first bars of a chart are often highly volatile and omission of these
                initial bars often leads to a better overall fit. Recommended range: 5-25
            lag: lag for crossover detection. Lower values result in earlier crossover, Recommended range: 1-2
        """

        # Estimations
        yhat1 = self.kernel_regression(close, h, r, x0)
        yhat2 = self.kernel_regression(close, h - lag, r, x0)

        # Rates of Change
        was_bearish = yhat1[2] > yhat1[1]
        was_bullish = yhat1[2] < yhat1[1]
        is_bearish = yhat1[1] > yhat1[0]
        is_bullish = yhat1[1] < yhat1[0]
        is_bearish_change = is_bearish and was_bullish
        is_bullish_change = is_bullish and was_bearish

        # Crossover
        is_bullish_cross = self.crossup(yhat2, yhat1)
        is_bearish_cross = self.crossdown(yhat2, yhat1)
        # is_bullish_smooth = yhat2 > yhat1
        # is_bearish_smooth = yhat2 < yhat1

        # Signal
        if is_bullish_cross or is_bullish_change:
            return 1
        elif is_bearish_cross or is_bearish_change:
            return -1
        else:
            return 0

    def kernel_regression(self, close: pd.Series, h: float, r: float, x0: int):
        """
        Nadaraya-Watson kernel regression
        options: Rational Quadratic Kernel / Gaussian Kernel / Periodic Kernel / Locally Periodic Kernel 
        """
        current_weight = 0
        cumulative_weight = 0
        size = len(close)
        estimates = np.empty(size - x0)
        i = 0
        while i + x0 < size:
            y = close.iloc[i]
            w = np.power(1 + (np.power(i, 2) / ((np.power(h, 2) * 2 * r))), -r)
            current_weight += y * w
            cumulative_weight += w
            estimates[i] = current_weight / cumulative_weight
            i += 1
        return estimates

    def crossup(self, a, b):
        """
        a cross up b from the bottom
        """
        return a[-1] > b[-1] and a[-2] < b[-2]

    def crossdown(self, a, b):
        """
        a cross down b from the top
        """
        return a[-1] < b[-1] and a[-2] > b[-2]
