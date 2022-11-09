

from utils.utils import Interval, max_step_by_day, get_auth

from .base_trader import BaseTrader

class FugureTrader(BaseTrader):
    def __init__(self, account = "a1"):
        self.auth = get_auth(account=account)