from datetime import date, datetime
from contextlib import closing


import wandb
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask, BacktestFinished, TqSim
from tqsdk.objs import Quote
from tqsdk.ta import EXPMA

import pytz


def close_pos_by_day(dt: int, current_pos: int):
    """
    close position at the end of the day
    """
    exchange_tz = pytz.timezone('Asia/Shanghai')
    dt = datetime.utcfromtimestamp(dt / 1e9).astimezone(exchange_tz)
    if (dt.hour == 14 and dt.minute > 55) or (dt.hour == 22 or dt.minute > 55):
        return 0
    return current_pos


def backtest(
    auth: TqAuth,
    commodity: str,
    symbol: str,
    is_wandb: bool,
    commission_fee: float,
    volume: int,
    start_dt=date(2022, 1, 1),
    end_dt=date(2022, 8, 1)
):
    sim = TqSim(init_balance=200000)
    api = TqApi(account=sim, auth=auth, backtest=TqBacktest(
        start_dt=start_dt, end_dt=end_dt), web_gui=False)
    if is_wandb:
        wandb.init(project="backtest-1",
                   config={"commodity": commodity, "symbol": symbol, "factor": "ema"})
    with closing(api):
        try:
            print("Backtesting")
            instrument_quote: Quote = api.get_quote(symbol)
            target_post = TargetPosTask(
                api, instrument_quote.underlying_symbol)
            klines = api.get_kline_serial(
                symbol, duration_seconds=60, data_length=200)
            account = api.get_account()
            current_pos = 0
            threshold_ratio = 0.0
            while True:
                api.wait_update()
                if api.is_changing(instrument_quote, "underlying_symbol"):
                    print("changing underlying_symbol")
                    target_post.set_target_volume(0)
                    target_post = TargetPosTask(
                        api, instrument_quote.underlying_symbol)
                    target_post.set_target_volume(current_pos)
                    sim.set_commission(
                        symbol=instrument_quote.underlying_symbol, commission=commission_fee)
                    print("underlying_symbol changed to",
                          instrument_quote.underlying_symbol)

                if api.is_changing(klines):
                    expma = EXPMA(klines, 12, 50)
                    ema_fast = expma['ma1']
                    ema_slow = expma['ma2']
                    pre_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
                    post_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2]
                    if pre_diff < threshold_ratio * klines.iloc[-1].close and post_diff > threshold_ratio * klines.iloc[-1].close:
                        # print("buy")
                        current_pos = volume
                    elif pre_diff > -threshold_ratio * klines.iloc[-1].close and post_diff < -threshold_ratio * klines.iloc[-1].close:
                        # print("sell")
                        current_pos = -volume

                    current_pos = close_pos_by_day(klines.iloc[-1].datetime)
                    target_post.set_target_volume(current_pos)

                    if is_wandb:
                        wandb.log({
                            "close_price": klines.iloc[-1].close,
                            "static_balance": account.static_balance,
                            "account_balance": account.balance,
                            "commission": account.commission,
                            "position": current_pos,
                        })
        except BacktestFinished:
            print("Backtest done")
