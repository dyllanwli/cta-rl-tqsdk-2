from datetime import date


from utils.utils import get_auth
from utils.dataloader import get_symbols_by_names


class FugureTrader:
    def __init__(self, account="a1"):
        self.auth, _ = get_auth(account)
        self.commodity = "iron_orb"
        self.symbol = get_symbols_by_names([self.commodity])[0]
        self.is_wandb = True
        self.volume = 5
        self.commission_fee = 7.7

    def backtest(self, strategy: str = "simple_ema"):
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
            tick_price = 0.5
            close_countdown_second = 3
            backtest(
                auth=self.auth,
                symbol=symbol,
                is_wandb=self.is_wandb,
                commission_fee=self.commission_fee,
                volume=self.volume,
                tick_price=tick_price,
                close_countdown_second=close_countdown_second,
                start_dt=date(2022, 11, 20),
                end_dt=date(2022, 11, 30)
            )
