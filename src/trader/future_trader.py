
from datetime import date
from contextlib import closing


from utils.utils import Interval, max_step_by_day, get_auth
from utils.dataloader import get_symbols_by_names
from tqsdk import TqApi, TqBacktest, TargetPosTask, BacktestFinished
from tqsdk.ta import EXPMA
from tqsdk.objs import Quote

import wandb

class FugureTrader:
    def __init__(self, account = "a2"):
        self.auth, _ = get_auth(account)
        self.commodity = "methanol"
        self.symbol = get_symbols_by_names([self.commodity])[0]
        self.current_pos = 0
        self.is_wandb = True
        self.threshold = 0.0
        self.volume = 5


    def backtest(self, start_dt=date(2020, 1, 1), end_dt=date(2022, 8, 1)):
        api = TqApi(auth=self.auth, backtest=TqBacktest(start_dt=start_dt, end_dt=end_dt))
        instrument_quote: Quote = api.get_quote(self.symbol)
        target_post = TargetPosTask(api, instrument_quote.underlying_symbol)
        klines = api.get_kline_serial(self.symbol, duration_seconds=60, data_length=200)
        account = api.get_account()
        print("Backtesting")
        if self.is_wandb:
            wandb.init(project="backtest-1",  config={"commodity": self.commodity, "symbol": self.symbol, "factor": "ema"})
        with closing(api):
            try:
                while True:
                    api.wait_update()
                    if api.is_changing(instrument_quote, "underlying_symbol"):
                        print("changing underlying_symbol")
                        target_post.set_target_volume(0)
                        target_post = TargetPosTask(api, instrument_quote.underlying_symbol)
                        target_post.set_target_volume(self.current_pos)
                        print("underlying_symbol changed to", instrument_quote.underlying_symbol)
                    
                    if api.is_changing(klines):
                        expma = EXPMA(klines, 12, 50)
                        ema_fast = expma['ma1']
                        ema_slow = expma['ma2']
                        pre_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
                        post_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2]
                        if pre_diff > self.threshold * klines.iloc[-1].close and post_diff <= self.threshold * klines.iloc[-1].close:
                            # print("buy")
                            target_post.set_target_volume(self.volume)
                            self.current_pos = self.volume
                        elif pre_diff < -self.threshold * klines.iloc[-1].close and post_diff >= -self.threshold * klines.iloc[-1].close:
                            # print("sell")
                            target_post.set_target_volume(-self.volume)
                            self.current_pos = -self.volume

                        if self.is_wandb:
                            wandb.log({
                                "klines": klines.iloc[-1].close, 
                                "static_balance": account.static_balance,
                                "account_balance": account.balance,
                                "commission": account.commission,
                                "ema_fast": ema_fast.iloc[-1],
                                "ema_slow": ema_slow.iloc[-1],
                            })
            except BacktestFinished:
                print("Backtest done")
                

