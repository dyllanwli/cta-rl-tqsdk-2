from datetime import date, datetime
from contextlib import closing


import wandb
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask, BacktestFinished, TqSim
from tqsdk.objs import Quote, Account
from tqsdk.tafunc import ema, ma
from tqsdk.ta import ATR
import pytz

import pandas as pd
import numpy as np


class HalfHull:
    def __init__(self, auth: TqAuth, commission_fee: float = 4.5, volume: int = 1, is_wandb: bool = True):
        self.auth = auth
        self.tick_price = 0.5
        self.commission_fee = commission_fee
        self.is_wandb = is_wandb
        self.volume = volume
        self.ema_periods = [5, 10, 20, 60]

    def backtest(
        self,
        symbol: str,
        start_dt=date(2022, 11, 1),
        end_dt=date(2022, 11, 30)
    ):
        assert self.tick_price > 0, "tick_price must be positive"
        sim = TqSim(init_balance=200000)
        self.api = TqApi(account=sim, auth=self.auth, backtest=TqBacktest(
            start_dt=start_dt, end_dt=end_dt), web_gui=False)
        sim.set_commission(symbol=symbol, commission=self.commission_fee)
        self.account: Account = self.api.get_account()

        print("Subscribing quote")
        quote: Quote = self.api.get_quote(symbol)

        klines_5m = self.api.get_kline_serial(
            symbol, duration_seconds=60, data_length=300)
        self.target_pos_task = TargetPosTask(self.api, symbol, price="ACTIVE")

        with closing(self.api):
            try:
                while True:
                    self.api.wait_update()
                    if self.check_trading_time(klines_5m['datetime'].iloc[-1]):
                        signal = self.get_signal(klines_5m)

                        if signal == 1:
                            self.target_pos_task.set_target_volume(self.volume)
                        elif signal == -1:
                            self.target_pos_task.set_target_volume(
                                -self.volume)
                        else:
                            self.close_position()
                        self.api.wait_update()
                    else:
                        self.close_position()

                    if self.is_wandb:
                        wandb.init(project="backtest-1",
                                   config={"symbol": symbol})

                    if self.is_wandb:
                        wandb.log({
                            "last_price": quote.last_price,
                            "static_balance": self.account.static_balance,
                            "account_balance": self.account.balance,
                            "commission": self.account.commission,
                        })
            except BacktestFinished:
                print("Backtest done")

    def EXPMA(self, data: pd.DataFrame, ema_periods: list) -> dict:
        """
        Calculate exponential moving average
        """
        ema_data = {}
        for period in ema_periods:
            ema_data[period] = ema(data.close, period)
        return ema_data

    def check_trading_time(self, dt: int):
        """
        close position at the end of the day
        """
        exchange_tz = pytz.timezone('Asia/Shanghai')
        dt = datetime.utcfromtimestamp(dt / 1e9).astimezone(exchange_tz)
        if (dt.hour == 14 and dt.minute > 55) or (dt.hour == 22 or dt.minute > 55):
            return False
        else:
            return True

    def close_position(self):
        self.target_pos_task.set_target_volume(0)

    def get_signal(self, klines_1m, klines_5s):
        """
        1: long
        -1: short
        0: close position
        """
        # get ema data
        ema_1m = self.EXPMA(klines_1m, self.ema_periods)
        ema_1s = self.EXPMA(klines_5s, self.ema_periods)

    def hull_butterfly_oscillator(self, klines_5m):
        """
        Hull Butterfly Oscillator
        """
        return

    def sma(self, series: pd.Series, l=14, columns=["high", "low"]):
        """
        Simple Moving Average
        """
        return series.rolling(l).mean()

    def halftrend(self, data: pd.DataFrame, atr_range = 100, amplitude = 2, deviation = 2):
        out = []

        trend = [0]
        next_trend = [0]
        up = 0
        down = 0
        atr_high = 0
        atr_low = 0

        atr2 = ATR(data, n = atr_range)["atr"] / 2
        deviation = deviation * atr2

        # get highest price of data.high within amplitude 
        highprice = max()

        # [high, close, low]
        for i in range(atr_range, len(data)+1):
            data_slice = data.iloc[i-atr_range:i]
            maxlow = data_slice[len(data_slice)-2][2]
            minhigh = data_slice[len(data_slice)-2][0]
            atr = ATR(data_slice, atr_range)["atr"]
            atr2 = atr2[len(atr2)-1] / 2
            dev = deviation * atr2
            
            highprice = max(data_slice[len(data_slice)-2][0], data_slice[len(data_slice)-1][0])
            lowprice = min(data_slice[len(data_slice)-2][2], data_slice[len(data_slice)-1][2])

            highs = list(map(lambda x: x[0], data_slice[len(
                data_slice)-amplitude:len(data_slice)]))
            lows = list(map(lambda x: x[2], data_slice[len(
                data_slice)-amplitude:len(data_slice)]))
            highma = self.sma(highs, amplitude)
            lowma = self.sma(lows, amplitude)

            if next_trend[len(next_trend)-1] == 1:
                maxlow = max(lowprice, maxlow)
                if highma[0] < maxlow and data_slice[len(data_slice)-1][1] < data_slice[len(data_slice)-2][2]:
                    trend.append(1)
                    next_trend.append(0)
                    minhigh = data_slice[len(data_slice)-2][0]
            else:
                minhigh = min(highprice, minhigh)
                if lowma[0] > minhigh and data_slice[len(data_slice)-1][1] < data_slice[len(data_slice)-2][0]:
                    trend.append(0)
                    next_trend.append(1)
                    maxlow = lowprice
            if trend[len(trend)-1] == 0:
                if not np.isnan(trend[len(trend)-2]) and trend[len(trend)-2] != 0:
                    if np.isnan(down[len(down)-2]):
                        up.append(down[len(down-1)])
                    else:
                        up.append(down[len(down)-2])
                else:
                    if np.isnan(up[len(up)-2]):
                        up.append(maxlow)
                    else:
                        up.append(max(up[len(up)-2], maxlow))
                direction = 'long'
                atrHigh = up[len(up)-1] + dev
                atrLow = up[len(up)-1] - dev
            else:
                if not np.isnan(trend[len(trend)-2] and trend[len(trend)-2] != 1):
                    if np.isnan(up[len(up)-2]):
                        down.append(up[len(up)-1])
                    else:
                        down.append(up[len(up)-2])
                else:
                    if np.isnan(down[len(down)-2]):
                        down.append(minhigh)
                    else:
                        down.append(min(minhigh, down[len(down)-2]))
                direction = 'short'
                atrHigh = down[len(down)-1] + dev
                atrLow = down[len(down)-1] - dev
            if trend[len(trend)-1] == 0:
                out.append([atrHigh, up[len(up)-1], atrLow, direction])
            else:
                out.append([atrHigh, down[len(down)-1], atrLow, direction])

        return out
