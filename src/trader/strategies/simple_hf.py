from datetime import date, datetime
from contextlib import closing

from collections import deque
import time
import wandb
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask, BacktestFinished, TqSim
from tqsdk.objs import Quote, Order
from tqsdk.ta import EXPMA

import pytz
import pandas as pd 


def close_pos_by_day(dt: int, current_pos: int):
    """
    close position at the end of the day
    """
    exchange_tz = pytz.timezone('Asia/Shanghai')
    dt = datetime.utcfromtimestamp(dt / 1e9).astimezone(exchange_tz)
    if (dt.hour == 14 and dt.minute > 55) or (dt.hour == 22 or dt.minute > 55):
        return 0
    return current_pos

def get_high_direction(bar: pd.DataFrame):
    """
    get direction with EMA
    """
    ema_5 = EXPMA(bar.close, 5)
    # if ema is going up, long
    if ema_5[-1] > ema_5[-2]:
        return 1
    # if ema is going down, short
    elif ema_5[-1] < ema_5[-2]:
        return -1
    else:
        return 0

def backtest(
    auth: TqAuth,
    symbol: str,
    is_wandb: bool,
    commission_fee: float,
    volume: int,
    tick_price: float,
    close_countdown_seconds: int,
    warmup_seconds: int = 10,
    start_dt=date(2022, 11, 1),
    end_dt=date(2022, 11, 30)
):  
    assert tick_price > 0, "tick_price must be positive"
    sim = TqSim(init_balance=200000)
    api = TqApi(account=sim, auth=auth, backtest=TqBacktest(
        start_dt=start_dt, end_dt=end_dt), web_gui=False)
    sim.set_commission(symbol=symbol, commission_fee=commission_fee)
    if is_wandb:
        wandb.init(project="backtest-1",
                   config={"symbol": symbol, "factor": "ema"})

    with closing(api):
        try:
            print("Backtesting")
            high_freq_bar = api.get_kline_serial(
                symbol, duration_seconds=1, data_length=200)
            quote: Quote = api.get_quote(symbol)
            account = api.get_account()
            current_pos = 0

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

                if api.is_changing(high_freq_bar.iloc[-1], "datetime"):
                    high_direction = get_high_direction(high_freq_bar)
                    if high_direction == 1:
                        open_order: Order = api.insert_order(
                            symbol=symbol, direction="BUY", offset="OPEN", volume=volume, limit_price = quote.ask_price1, advanced="FAK")

                        while open_order.status != "FINISHED":
                            api.wait_update()

                        close_price = open_order.trade_price + tick_price
                        close_price = max(close_price, quote.bid_price1)
                        close_volume = volume - open_order.volume_left
                        if close_volume > 0:
                            close_order: Order = api.insert_order(
                                symbol=symbol, direction="SELL", offset="CLOSE", volume=close_volume, limit_price=close_price)
                            close_time = time.time()
                            while close_order.status != "FINISHED" and api.is_changing(quote):
                                historical_ask_volume1.append(quote.ask_volume1)
                                historical_bid_volume1.append(quote.bid_volume1)
                                ask_bid_index = sum(historical_ask_volume1) / sum(historical_bid_volume1)

                                api.wait_update()
                                if time.time() - close_time >= close_countdown_seconds:
                                    print("cancel and reinsert order at market price")
                                    api.cancel_order(close_order)
                                    if (not close_order and api.is_changing(quote)):
                                        close_order: Order = api.insert_order(
                                            symbol=symbol, direction="SELL", offset="CLOSE", volume=close_order.get("volume_left", 0), limit_price=quote.bid_price1)
                                        close_time = time.time()
                                
                elif high_direction == -1:
                        open_order: Order = api.insert_order(
                            symbol=symbol, direction="SELL", offset="OPEN", volume=volume, limit_price = quote.bid_price1, advanced="FAK")

                        while open_order.status != "FINISHED":
                            api.wait_update()

                        close_price = open_order.trade_price - tick_price
                        close_volume = volume - open_order.volume_left
                        close_order: Order = api.insert_order(
                            symbol=symbol, direction="BUY", offset="CLOSE", volume=close_volume, limit_price=close_price)


                if is_wandb:
                    wandb.log({
                        "close_price": high_freq_bar.iloc[-1].close,
                        "static_balance": account.static_balance,
                        "account_balance": account.balance,
                        "commission": account.commission,
                        "position": current_pos,
                    })
        except BacktestFinished:
            print("Backtest done")
