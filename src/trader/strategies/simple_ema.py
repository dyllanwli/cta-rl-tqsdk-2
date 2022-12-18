from datetime import date, datetime
from contextlib import closing


import wandb
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask, BacktestFinished, TqSim
from tqsdk.objs import Quote, Account
from tqsdk.tafunc import ema
import pytz

import pandas as pd

class SimpleHFEMA:
    def __init__(self, auth: TqAuth, commission_fee: float = 4.5, volume: int = 1, is_wandb: bool = True):
        self.auth = auth
        self.tick_price = 0.5
        self.commission_fee = commission_fee
        self.is_wandb = is_wandb    
        self.volume = volume
        self.ema_periods = [5, 10, 20, 40, 60]

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
        klines_1m = self.api.get_kline_serial(symbol, duration_seconds=60, data_length=300)
        klines_1s = self.api.get_kline_serial(symbol, duration_seconds=5, data_length=300)
        self.target_pos_task = TargetPosTask(self.api, symbol, price="ACTIVE")

        with closing(self.api):
            try:
                while True:
                    self.api.wait_update()
                    if self.check_trading_time(klines_1m['datetime'].iloc[-1]):

                        ema_data = self.EXPMA(klines_1m, self.ema_periods)
                        signal = self.get_signal(klines_1m, ema_data)
                        if signal == 1:
                            self.target_pos_task.set_target_volume(self.volume)
                        elif signal == -1:
                            self.target_pos_task.set_target_volume(-self.volume)
                        else:
                            self.close_position()
                        self.api.wait_update()
                    else:
                        self.close_position()

                    if self.is_wandb:
                        wandb.init(project="backtest-1", config={"symbol": symbol})

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
    
    def get_signal(self, data, ema_data):
        """
        1: long
        -1: short
        0: close position
        """
        ma_0 = ema_data[self.ema_periods[0]]
        ma_1 = ema_data[self.ema_periods[1]]
        # if ma_0 crossover ma_1, long
        if ma_0.iloc[-2] < ma_1.iloc[-2] and ma_0.iloc[-1] > ma_1.iloc[-1]:
            return 1
        