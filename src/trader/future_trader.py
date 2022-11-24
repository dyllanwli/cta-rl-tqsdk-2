
from datetime import date
from contextlib import closing


from utils.utils import Interval, max_step_by_day, get_auth
from utils.dataloader import get_symbols_by_names
from tqsdk import TqApi, TqBacktest, TargetPosTask
from tqsdk.ta import EXPMA

import wandb

class FugureTrader:
    def __init__(self, account = "a3"):
        self.auth, _ = get_auth(account)
        self.commodity = "methanol"
        self.symbol = get_symbols_by_names([self.commodity])[0]
        self.current_pos = 0
        self.is_wandb = True
        if self.is_wandb:
            wandb.init(project="backtest-1",  config={"commodity": self.commodity, "symbol": self.symbol, "factor": "ema"})
        self.threshold = 0.01


    def backtest(self, start_dt=date(2020, 1, 1), end_dt=date(2022, 8, 1)):
        print("Backtesting")
        api = TqApi(auth=self.auth, backtest=TqBacktest(start_dt=start_dt, end_dt=end_dt))
        instrument_quote = api.get_quote(self.symbol)
        underlying_symbol = instrument_quote.underlying_symbol
        target_post = TargetPosTask(api, underlying_symbol)
        klines = api.get_kline_serial(self.symbol, duration_seconds=60, data_length=200)
        account = api.get_account()
        # factors
        expma = EXPMA(klines, 5, 20)
        with closing(api):
            try:
                while True:
                    while not api.is_changing(instrument_quote, "datetime"):
                        # wait for next trading day
                        api.wait_update()
                    while True:
                        api.wait_update()
                        if api.is_changing(instrument_quote, "underlying_symbol"):
                            target_post.set_target_volume(0)
                            target_post = TargetPosTask(api, instrument_quote.underlying_symbol)
                            target_post.set_target_volume(self.current_pos)
                            print("underlying_symbol changed")
                        
                        if api.is_changing(klines):
                            ema_fast = expma['ma1']
                            ema_slow = expma['ma2']
                            pre_diff = ema_fast[-1] - ema_slow[-1]
                            post_diff = ema_fast[-2] - ema_slow[-2]
                            if pre_diff > self.threshold * klines[-1].close and post_diff < self.threshold * klines[-1].close:
                                # buy
                                target_post.set_target_volume(1)
                                self.current_pos = 1
                            elif pre_diff < -self.threshold * klines[-1].close and post_diff > -self.threshold * klines[-1].close:
                                # sell
                                target_post.set_target_volume(-1)
                                self.current_pos = -1

                            if self.is_wandb:
                                wandb.log({
                                    "datetime": instrument_quote.datetime, 
                                    "klines": klines[-1].close, 
                                    "ema_fast": ema_fast, 
                                    "ema_slow": ema_slow, 
                                    "static_balance": account.static_balance,
                                    "account_balance": account.balance,
                                    "commission": account.commission,
                                })
            except Exception as e:
                print(e)
                print("Backtest done")
                

