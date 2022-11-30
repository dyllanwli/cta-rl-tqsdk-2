from datetime import date, datetime
from contextlib import closing


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
        return "BUY"
    # if ema is going down, short
    elif ema_5[-1] < ema_5[-2]:
        return "SELL"
    else:
        return 0

def backtest(
    auth: TqAuth,
    symbol: str,
    is_wandb: bool,
    commission_fee: float,
    volume: int,
    tick_price: float,
    start_dt=date(2022, 11, 1),
    end_dt=date(2022, 11, 30)
):
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

            account = api.get_account()
            current_pos = 0
            while True:
                api.wait_update()
                if api.is_changing(high_freq_bar.iloc[-1], "datetime"):
                    high_direction = get_high_direction(high_freq_bar)
                    if high_direction == "BUY":
                        order: Order = api.insert_order(
                            symbol=symbol, direction=high_direction, offset="OPEN", volume=volume, advanced="FAK")
                        if order.status == "FINISHED":
                            order_price = order.trade_price

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
