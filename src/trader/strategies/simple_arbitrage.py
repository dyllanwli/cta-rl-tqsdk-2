from datetime import date, datetime
from contextlib import closing
from typing import List, Tuple, Union

from collections import deque
import time
import wandb
from tqsdk import TqApi, TqAuth, TqBacktest, TargetPosTask, BacktestFinished, TqSim
from tqsdk.tafunc import time_to_s_timestamp
from tqsdk.objs import Quote, Order
from tqsdk.ta import EMA

import pytz
import pandas as pd


class SimpleArbitrage:
    def __init__(
        self,
        auth: TqAuth,
        is_wandb: bool = True,
        arbitrage_A: str = "SHFE.rb",
        arbitrage_B: str = "SHFE.hc",
        commission_fee_A: float = 3.7,
        commission_fee_B: float = 3.9,
        max_vol_ratio: float = 0.1,
        max_holding_vol: int = 20,
        start_dt=date(2022, 11, 1),
        end_dt=date(2022, 11, 30)
    ):
        sim = TqSim(init_balance=200000)
        self.api: TqApi = TqApi(account=sim, auth=auth, backtest=TqBacktest(
            start_dt=start_dt, end_dt=end_dt), web_gui=False)

        # get quote
        quote_A = self.api.get_quote("KQ.m@" + arbitrage_A)
        quote_B = self.api.get_quote("KQ.m@" + arbitrage_B)
        self.arbitrage_A, self.arbitrage_B = quote_A.underlying_symbol, quote_B.underlying_symbol
        # set commission fee
        sim.set_commission(symbol=arbitrage_A, commission=commission_fee_A)
        sim.set_commission(symbol=arbitrage_A, commission=commission_fee_B)
        if is_wandb:
            wandb.init(project="backtest-1",
                       config={"method": "arbitrage", "symbols": [arbitrage_A, arbitrage_B]})

        self.max_vol_ratio = max_vol_ratio
        self.max_holding_vol = max_holding_vol

    def backtest(self, ):
        # get bar
        bars = self.api.get_kline_serial(
            [self.arbitrage_A, self.arbitrage_B], 60, data_length=100)

        # get account
        quotes_A = self.api.get_quote(self.arbitrage_A)
        quotes_B = self.api.get_quote(self.arbitrage_B)

        # get account
        account = self.api.get_account()
        self.pos_A = self.api.get_position(self.arbitrage_A)
        self.pos_B = self.api.get_position(self.arbitrage_B)

        # get prices
        pos_arbitrage = 5
        neg_arbitrage = 5
        stop_loss_pos_arbitrage = 10  # 正套止损
        stop_loss_neg_arbitrage = 10  # 反套止损
        stop_profit_pos_arbitrage = 20  # 正套止盈
        stop_profit_neg_arbitrage = 20  # 反套止盈

        self.order_list: List[Tuple] = []
        order_lock: bool = False
        is_stop_loss_pos = False
        is_stop_loss_neg = False
        is_stop_profit_pos = False
        is_stop_profit_neg = False

        # target_pos_A = TargetPosTask(self.api, self.arbitrage_A)
        # target_pos_B = TargetPosTask(self.api, self.arbitrage_B)

        with closing(self.api):
            while True:
                self.api.wait_update()
                order_lock = self.is_order_alive()
                if order_lock:
                    # check if order is alive
                    continue
                # check if stop loss
                if is_stop_loss_pos or is_stop_loss_neg:
                    if is_stop_loss_pos and self.pos_A.pos_short == 0:
                        print("position stop loss end")
                        is_stop_loss_pos = False
                    elif is_stop_loss_neg and self.pos_A.pos_long == 0:
                        print("position stop loss end")
                        is_stop_loss_neg = False
                # check if stop profit
                if is_stop_profit_pos or is_stop_profit_neg:
                    if is_stop_profit_pos and self.pos_A.pos_short == 0:
                        print("position stop profit end")
                        is_stop_profit_pos = False
                    elif is_stop_profit_neg and self.pos_A.pos_long == 0:
                        print("position stop profit end")
                        is_stop_profit_neg = False

                is_stop_loss = is_stop_loss_neg or is_stop_loss_pos
                is_stop_profit = is_stop_profit_neg or is_stop_profit_pos
                is_stop_all = is_stop_loss or is_stop_profit

                # check quote
                if self.api.is_changing(quotes_A) or self.api.is_changing(quotes_B):
                    quote_time = pd.Timestamp(quotes_A.datetime)

                    # close all before day end - 5 min
                    if (quote_time.hour == 14 and quote_time.minute >= 55) or (quote_time.hour == 22 and quote_time.minute >= 55):
                        self.close_all(vol_per_trade)
                        continue

                    vol_per_trade = int(
                        min(quotes_A.ask_volume1, quotes_B.bid_volume1) * self.max_vol_ratio)

                    if vol_per_trade == 0:
                        continue
                    if (is_stop_loss or is_stop_profit) and (self.pos_A.pos_long == 0) and (self.pos_A.pos_short > 0):
                        print("positive stop profit or stop loss")
                        if self.pos_A.pos_short < vol_per_trade:
                            curr_vol = self.pos_A.pos_short
                        else:
                            curr_vol = vol_per_trade
                        if not curr_vol:
                            continue

                        order_A = self.api.insert_order(
                            symbol=self.arbitrage_A, direction="BUY", offset="CLOSE", volume=curr_vol)
                        order_B = self.api.insert_order(
                            symbol=self.arbitrage_B, direction="SELL", offset="CLOSE", volume=curr_vol)
                        self.order_list.append((order_A, order_B))
                        continue
                    elif (is_stop_loss or is_stop_profit) and (self.pos_A.pos_short == 0) and (self.pos_A.pos_long > 0):
                        print("negative stop profit or stop loss")
                        if self.pos_A.pos_long < vol_per_trade:
                            curr_vol = self.pos_A.pos_long
                        else:
                            curr_vol = vol_per_trade
                        if not curr_vol:
                            continue
                        order_A = self.api.insert_order(
                            symbol=self.arbitrage_A, direction="SELL", offset="CLOSE", volume=curr_vol)
                        order_B = self.api.insert_order(
                            symbol=self.arbitrage_B, direction="BUY", offset="CLOSE", volume=curr_vol)
                        self.order_list.append((order_A, order_B))
                        continue
                    elif is_stop_loss_pos and (not is_stop_loss) and (self.pos_A.pos_short == 0) and (self.pos_A.pos_long < self.max_holding_vol):
                        print("positive open")
                        order_A = self.api.insert_order(
                            symbol=self.arbitrage_A, direction="BUY", offset="OPEN", volume=vol_per_trade)
                        order_B = self.api.insert_order(
                            symbol=self.arbitrage_B, direction="SELL", offset="OPEN", volume=vol_per_trade)
                        self.order_list.append((order_A, order_B))
                        continue
                    elif is_stop_loss_neg and (not is_stop_loss) and (self.pos_A.pos_long == 0) and (self.pos_A.pos_short < self.max_holding_vol):
                        print("negative open")
                        order_A = self.api.insert_order(
                            symbol=self.arbitrage_A, direction="SELL", offset="OPEN", volume=vol_per_trade)
                        order_B = self.api.insert_order(
                            symbol=self.arbitrage_B, direction="BUY", offset="OPEN", volume=vol_per_trade)
                        self.order_list.append((order_A, order_B))
                        continue

                    # check stop loss and stop profit per minute
                    if quote_time.second == 0:
                        bar_slice = bars.iloc[-1]
                        spread = bar_slice.close - bar_slice.close1
                        if spread < stop_profit_neg_arbitrage and (not is_stop_profit_pos) and (not is_stop_loss):
                            print("negative stop profit filled")
                            if self.pos_A.pos_short >= 0 and (self.pos_A.pos_long) <= 0 and (self.pos_A.pos_short < self.max_holding_vol):
                                print(
                                    "hold A short and position less than max holding vol")
                                is_stop_profit_neg = True
                                order_A = self.api.insert_order(
                                    symbol=self.arbitrage_A, direction="SELL", offset="OPEN", volume=vol_per_trade)
                                order_B = self.api.insert_order(
                                    symbol=self.arbitrage_B, direction="BUY", offset="OPEN", volume=vol_per_trade)
                                self.order_list.append((order_A, order_B))
                                continue
                            elif self.pos_A.pos_long > 0:
                                print("hold A long")
                                is_stop_profit_neg = True
                                if self.pos_A.pos_long < vol_per_trade:
                                    curr_vol = self.pos_A.pos_long
                                else:
                                    curr_vol = vol_per_trade
                                if not curr_vol:
                                    continue
                                order_A = self.api.insert_order(
                                    symbol=self.arbitrage_A, direction="SELL", offset="CLOSE", volume=curr_vol)
                                order_B = self.api.insert_order(
                                    symbol=self.arbitrage_B, direction="BUY", offset="CLOSE", volume=curr_vol)
                                self.order_list.append((order_A, order_B))
                                continue
                            elif self.pos_A.pos_short >= self.max_holding_vol:
                                print(
                                    "hold A short and position more than max holding vol")
                                is_stop_profit_neg = False
                                continue
                        elif spread > stop_profit_pos_arbitrage and (not is_stop_profit_neg) and (not is_stop_loss):
                            print("positive stop profit filled")
                            if self.pos_A.pos_long >= 0 and (self.pos_A.pos_short) <= 0 and (self.pos_A.pos_long < self.max_holding_vol):
                                print(
                                    "hold A long and position less than max holding vol")
                                is_stop_profit_pos = True
                                order_A = self.api.insert_order(
                                    symbol=self.arbitrage_A, direction="BUY", offset="OPEN", volume=vol_per_trade)
                                order_B = self.api.insert_order(
                                    symbol=self.arbitrage_B, direction="SELL", offset="OPEN", volume=vol_per_trade)
                                self.order_list.append((order_A, order_B))
                                continue
                            elif self.pos_A.pos_short > 0:
                                print("hold A short")
                                is_stop_profit_pos = True
                                if self.pos_A.pos_short < vol_per_trade:
                                    curr_vol = self.pos_A.pos_short
                                else:
                                    curr_vol = vol_per_trade
                                if not curr_vol:
                                    continue
                                order_A = self.api.insert_order(
                                    symbol=self.arbitrage_A, direction="BUY", offset="CLOSE", volume=curr_vol)
                                order_B = self.api.insert_order(
                                    symbol=self.arbitrage_B, direction="SELL", offset="CLOSE", volume=curr_vol)
                                self.order_list.append((order_A, order_B))
                                continue
                            elif self.pos_A.pos_long >= self.max_holding_vol:
                                print(
                                    "hold A long and position more than max holding vol")
                                is_stop_profit_pos = False
                                continue
                        elif spread > stop_loss_pos_arbitrage and (self.pos_A.pos_short > 0) and (not is_stop_profit):
                            print("positive stop loss filled and not stop profit")
                            is_stop_loss_pos = True
                            if self.pos_A.pos_short < vol_per_trade:
                                curr_vol = self.pos_A.pos_short
                            else:
                                curr_vol = vol_per_trade
                            if not curr_vol:
                                continue
                            order_A = self.api.insert_order(
                                symbol=self.arbitrage_A, direction="BUY", offset="CLOSE", volume=curr_vol)
                            order_B = self.api.insert_order(
                                symbol=self.arbitrage_B, direction="SELL", offset="CLOSE", volume=curr_vol)
                            self.order_list.append((order_A, order_B))
                            continue
                        elif spread < stop_loss_neg_arbitrage and (self.pos_A.pos_long > 0) and (not is_stop_profit):
                            print("negative stop loss filled and not stop profit")
                            is_stop_loss_neg = True
                            if self.pos_A.pos_long < vol_per_trade:
                                curr_vol = self.pos_A.pos_long
                            else:
                                curr_vol = vol_per_trade
                            if not curr_vol:
                                continue
                            order_A = self.api.insert_order(
                                symbol=self.arbitrage_A, direction="SELL", offset="CLOSE", volume=curr_vol)
                            order_B = self.api.insert_order(
                                symbol=self.arbitrage_B, direction="BUY", offset="CLOSE", volume=curr_vol)
                            self.order_list.append((order_A, order_B))
                            continue
                    spread_pos = abs(
                        quotes_A["bid_price1"] - quotes_B["ask_price1"])
                    spread_neg = abs(
                        quotes_A["ask_price1"] - quotes_B["bid_price1"])
                    if spread_pos >= pos_arbitrage and (not is_stop_all):
                        print("positive filled and not stop all")
                        if self.pos_A.pos_long > 0:
                            print("hold long when positive filled, close long")
                            if self.pos_A.pos_long < vol_per_trade:
                                curr_vol = self.pos_A.pos_long
                            else:
                                curr_vol = vol_per_trade
                            if not curr_vol:
                                continue
                            order_A = self.api.insert_order(
                                symbol=self.arbitrage_A, direction="SELL", offset="CLOSE", volume=curr_vol)
                            order_B = self.api.insert_order(
                                symbol=self.arbitrage_B, direction="BUY", offset="CLOSE", volume=curr_vol)
                            self.order_list.append((order_A, order_B))
                            continue
                        elif self.pos_A.pos_short < self.max_holding_vol:
                            print(
                                "no long positive posiiton and short position less than max holding vol")
                            order_A = self.api.insert_order(
                                symbol=self.arbitrage_A, direction="BUY", offset="OPEN", volume=vol_per_trade)
                            order_B = self.api.insert_order(
                                symbol=self.arbitrage_B, direction="SELL", offset="OPEN", volume=vol_per_trade)
                            self.order_list.append((order_A, order_B))
                            continue
                    if spread_neg <= neg_arbitrage and (not is_stop_all):
                        print("negative filled and not stop all")
                        if self.pos_A.pos_short > 0:
                            print("hold short when negative filled, close short")
                            if self.pos_A.pos_short < vol_per_trade:
                                curr_vol = self.pos_A.pos_short
                            else:
                                curr_vol = vol_per_trade
                            if not curr_vol:
                                continue
                            order_A = self.api.insert_order(
                                symbol=self.arbitrage_A, direction="BUY", offset="CLOSE", volume=curr_vol)
                            order_B = self.api.insert_order(
                                symbol=self.arbitrage_B, direction="SELL", offset="CLOSE", volume=curr_vol)
                            self.order_list.append((order_A, order_B))
                            continue

    def is_order_alive(self):
        for order_tuple in self.order_list:
            if self.api.is_changing(
                order_tuple[0], ["status", "volume_left"]
            ) or self.api.is_changing(order_tuple[1], ["status", "volume_left"]):
                if (order_tuple[0].volume_left == 0) and (order_tuple[1].volume_left == 0):
                    self.order_list.remove(order_tuple)
        if not self.order_list:
            return False
        return True

    def close_all(self, vol_per_trade: int = 0) -> bool:
        """
        close all position before end of day
        """
        if self.pos_A.pos_long > 0:
            if self.pos_A.pos_long < vol_per_trade:
                curr_vol = self.pos_A.pos_long
            else:
                curr_vol = vol_per_trade
            if not curr_vol:
                return
            order_A = self.api.insert_order(
                symbol=self.arbitrage_A,
                direction="SELL",
                offset="CLOSETODAY",
                volume=curr_vol,
            )
            order_B = self.api.insert_order(
                symbol=self.arbitrage_B,
                direction="BUY",
                offset="CLOSETODAY",
                volume=curr_vol,
            )
            self.order_list.append((order_A, order_B))
            return True
        if self.pos_A.pos_short > 0:
            if self.pos_A.pos_short < vol_per_trade:
                curr_vol = self.pos_A.pos_short
            else:
                curr_vol = vol_per_trade
            if not curr_vol:
                return False
            order_A = self.api.insert_order(
                symbol=self.arbitrage_A,
                direction="BUY",
                offset="CLOSETODAY",
                volume=curr_vol,
            )
            order_B = self.api.insert_order(
                symbol=self.arbitrage_B,
                direction="SELL",
                offset="CLOSETODAY",
                volume=curr_vol,
            )
            self.order_list.append((order_A, order_B))
            return True
        return False
