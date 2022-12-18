from datetime import date, datetime
from contextlib import closing

from collections import deque
import time
import wandb
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask, BacktestFinished, TqSim
from tqsdk.tafunc import time_to_s_timestamp
from tqsdk.objs import Quote, Order, Account
from tqsdk.ta import EMA

import pytz
import pandas as pd
class SimpleHFAroon:
    """
    Using Aroon indicator to determine the trend
    """
    def __init__(self, auth: TqAuth, commission_fee: float = 4.5, volume: int = 1, is_wandb: bool = True):
        self.auth = auth
        self.tick_price = 0.5
        self.commission_fee = commission_fee
        self.is_wandb = is_wandb    
        self.volume = volume

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
        bar_data = self.api.get_kline_serial(symbol, duration_seconds=60, data_length=300)
        self.target_pos_task = TargetPosTask(self.api, symbol, price="PASSIVE")

        with closing(self.api):
            try:
                while True:
                    self.api.wait_update()
                    if self.check_trading_time(bar_data['datetime'].iloc[-1]):
                        up, down = self.aroon(bar_data, lb=30)
                        if up.iloc[-1] > 80 and down.iloc[-1] < 20:
                            self.target_pos_task.set_target_volume(self.volume)
                        elif up.iloc[-1] < 20 and down.iloc[-1] > 80:
                            self.target_pos_task.set_target_volume(-self.volume)
                        else:
                            self.target_pos_task.set_target_volume(0)
                        self.api.wait_update()
                    else:
                        self.close_position()

                
                    if self.is_wandb:
                        wandb.init(project="backtest-1", config={"symbol": symbol})

                    if self.is_wandb:
                        wandb.log({
                            "last_price": quote.last_price,
                            "up": up.iloc[-1],
                            "down": down.iloc[-1],
                            "static_balance": self.account.static_balance,
                            "account_balance": self.account.balance,
                            "commission": self.account.commission,
                        })
            except BacktestFinished:
                print("Backtest done")


    def aroon(self, data, lb=25):

        df = data.copy()
        df['up'] = 100 * df['high'].rolling(lb + 1).apply(lambda x: x.argmax()) / lb
        df['down'] = 100 * df['low'].rolling(lb + 1).apply(lambda x: x.argmin()) / lb

        return df['up'], df['down'] 

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
        

    
