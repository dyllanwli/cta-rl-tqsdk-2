from datetime import date


from utils.utils import get_auth
from utils.dataloader import get_symbols_by_names


class FugureTrader:
    def __init__(self, account="a4"):
        self.auth, _ = get_auth(account)
        self.commodity = "iron_orb"
        self.symbol = get_symbols_by_names([self.commodity])[0]
        self.is_wandb = True
        self.volume = 5
        self.commission_fee = 7.7

    def backtest(self, strategy: str = "simple_arbitrage"):
        if strategy == "simple_ema":
            from .strategies.simple_ema import backtest
            backtest(
                auth=self.auth,
                commodity=self.commodity,
                symbol=self.symbol,
                is_wandb=self.is_wandb,
                commission_fee=self.commission_fee,
                volume=self.volume,
                start_dt=date(2022, 1, 1),
                end_dt=date(2022, 8, 1)
            )
        elif strategy == "simple_hf":
            from .strategies.simple_hf import backtest
            symbol = "DCE.i2301"
            tick_price = 1
            close_countdown_seconds = 5
            backtest(
                auth=self.auth,
                symbol=symbol,
                is_wandb=self.is_wandb,
                commission_fee=self.commission_fee,
                volume=self.volume,
                tick_price=tick_price,
                close_countdown_seconds=close_countdown_seconds,
                start_dt=date(2022, 11, 20),
                end_dt=date(2022, 11, 30)
            )
        elif strategy == "simple_arbitrage":
            from .strategies.simple_arbitrage import SimpleArbitrage
            model = SimpleArbitrage(
                auth=self.auth,
            )
            model.backtest()
        elif strategy == "simple_hf_order_book":
            from .strategies.simple_hf_order_book import SimpleHFOrderBook
            symbol = "DCE.i2301"
            model = SimpleHFOrderBook(
                auth=self.auth,
            )
            model.backtest(
                symbol=symbol,
                start_dt=date(2022, 11, 1),
                end_dt=date(2022, 11, 30)
            )
