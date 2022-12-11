from datetime import date, datetime
from contextlib import closing

from collections import deque
import time
import wandb
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask, BacktestFinished, TqSim
from tqsdk.tafunc import time_to_s_timestamp
from tqsdk.objs import Quote, Order
from tqsdk.ta import EMA

import pytz
import pandas as pd


class SimpleHFOrderBook:
    def __init__(
            self,
            auth: TqAuth):
        sim = TqSim(init_balance=200000)
        self.auth = auth
        self.tick_price = 0.5
        self.is_wandb = False

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
        self.account = self.api.get_account()

        ticks: pd.DataFrame = self.api.get_tick_serial(symbol, data_length = 1000)
        
        if self.is_wandb:
            wandb.init(project="backtest-1", config={"symbol": symbol})


        if self.is_wandb:
            wandb.log({
                "static_balance": self.account.static_balance,
                "account_balance": self.account.balance,
                "commission": self.account.commission
            })
        

    def check_is_daytrading(dt: int):
        """
        close position at the end of the day
        """
        exchange_tz = pytz.timezone('Asia/Shanghai')
        dt = datetime.utcfromtimestamp(dt / 1e9).astimezone(exchange_tz)
        if (dt.hour == 14 and dt.minute > 55) or (dt.hour == 22 or dt.minute > 55):
            return True
        else:
            return False


def backtest(
    auth: TqAuth,
    symbol: str,
    is_wandb: bool,
    commission_fee: float,
    volume: int,
    tick_price: float,
    close_countdown_seconds: int,
    warmup_seconds: int = 1,
    start_dt=date(2022, 11, 1),
    end_dt=date(2022, 11, 30)
):
    assert tick_price > 0, "tick_price must be positive"
    sim = TqSim(init_balance=200000)
    api = TqApi(account=sim, auth=auth, backtest=TqBacktest(
        start_dt=start_dt, end_dt=end_dt), web_gui=False)
    sim.set_commission(symbol=symbol, commission=commission_fee)
    if is_wandb:
        wandb.init(project="backtest-1",
                   config={"symbol": symbol, "factor": "ema"})

    with closing(api):
        try:
            print("Backtesting")
            high_freq_bar = api.get_kline_serial(
                symbol, duration_seconds=1, data_length=20)
            quote: Quote = api.get_quote(symbol)
            account = api.get_account()

            historical_ask_volume1 = deque([], maxlen=warmup_seconds)
            historical_bid_volume1 = deque([], maxlen=warmup_seconds)
            # warmup to get historical data
            print("Warmup...")
            warmup_time = time.time()
            while api.is_changing(quote, "datetime") and time.time() - warmup_time < warmup_seconds:
                api.wait_update()
                historical_ask_volume1.append(quote.ask_volume1)
                historical_bid_volume1.append(quote.bid_volume1)

            while True:
                api.wait_update()

                if api.is_changing(quote) and check_daytrading(high_freq_bar.iloc[-1]["datetime"]):
                    historical_ask_volume1.append(quote.ask_volume1)
                    historical_bid_volume1.append(quote.bid_volume1)
                    bid_ask_diff = sum(historical_bid_volume1) - \
                        sum(historical_ask_volume1)
                    direction = 1 if bid_ask_diff > 0 else -1

                    if direction == 1:
                        open_order: Order = api.insert_order(
                            symbol=symbol, direction="BUY", offset="OPEN", volume=volume, limit_price=quote.ask_price1, advanced="FAK")

                        while open_order.status == "ALIVE":
                            api.wait_update()

                        close_price = open_order.trade_price + tick_price
                        close_price = max(close_price, quote.bid_price1)
                        close_volume = volume - open_order.volume_left
                        if close_volume > 0:
                            close_order: Order = api.insert_order(
                                symbol=symbol, direction="SELL", offset="CLOSE", volume=close_volume, limit_price=close_price)
                            close_time = time.time()
                            while close_order.status == "ALIVE" and close_order.volume_left != 0:
                                api.wait_update()

                                if api.is_changing(close_order):
                                    print("close_order: %s, completed: %d" % (
                                        close_order.status, close_order.volume_orign - close_order.volume_left))

                                if time.time() - close_time >= close_countdown_seconds:
                                    # cancel and reinsert order at market price
                                    if close_order.status == "ALIVE" and close_order.trade_price != quote.bid_price1:
                                        api.cancel_order(close_order)
                                        close_order: Order = api.insert_order(
                                            symbol=symbol, direction="SELL", offset="CLOSE", volume=close_order.volume_left, limit_price=quote.bid_price1)
                                    continue

                                if close_price < quote.bid_price1:
                                    # cancel the order and insert a new one with higher price
                                    close_price = quote.bid_price1
                                    api.cancel_order(close_order)

                                if api.is_changing(close_order) and close_order.volume_left != 0 and close_order.status == "FINISHED":
                                    # if the order is cancelled, insert a new one
                                    close_order: Order = api.insert_order(
                                        symbol=symbol, direction="SELL", offset="CLOSE", volume=close_order.volume_left, limit_price=close_price)

                    elif direction == -1:
                        open_order: Order = api.insert_order(
                            symbol=symbol, direction="SELL", offset="OPEN", volume=volume, limit_price=quote.bid_price1, advanced="FAK")

                        while open_order.status == "ALIVE":
                            api.wait_update()

                        close_price = open_order.trade_price - tick_price
                        close_price = min(close_price, quote.ask_price1)
                        close_volume = volume - open_order.volume_left
                        if close_volume > 0:
                            close_order: Order = api.insert_order(
                                symbol=symbol, direction="BUY", offset="CLOSE", volume=close_volume, limit_price=close_price)
                            close_time = time.time()
                            while close_order.status == "ALIVE" and close_order.volume_left != 0:
                                api.wait_update()

                                if api.is_changing(close_order):
                                    print("close_order: %s, completed: %d" % (
                                        close_order.status, close_order.volume_orign - close_order.volume_left))

                                if time.time() - close_time >= close_countdown_seconds:
                                    # cancel and reinsert order at market price
                                    if close_order.status == "ALIVE" and close_order.trade_price != quote.ask_price1:
                                        api.cancel_order(close_order)
                                        close_order: Order = api.insert_order(
                                            symbol=symbol, direction="BUY", offset="CLOSE", volume=close_order.volume_left, limit_price=quote.ask_price1)
                                    continue

                                if close_price > quote.ask_price1:
                                    # cancel the order and insert a new one with higher price
                                    close_price = quote.ask_price1
                                    api.cancel_order(close_order)

                                if api.is_changing(close_order) and close_order.volume_left != 0 and close_order.status == "FINISHED":
                                    # if the order is cancelled, insert a new one
                                    close_order: Order = api.insert_order(
                                        symbol=symbol, direction="BUY", offset="CLOSE", volume=close_order.volume_left, limit_price=close_price)

        except BacktestFinished:
            print("Backtest done")
