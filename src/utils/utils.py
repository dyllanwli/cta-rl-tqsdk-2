
from typing import NamedTuple
import yaml
import os.path

from tqsdk import TqApi, TqAuth, TqAccount

DIR = os.path.dirname(os.path.abspath(__file__))

def get_config() -> dict:
    # Read YAML file
    configFilePath = os.path.join(DIR, "config.yaml")
    with open(configFilePath, 'r') as stream:
        config = yaml.safe_load(stream)
        return config

SETTINGS = get_config()


def get_auth(account: str = 'a1', live_account: str = None):
    # Get API
    account_settings = SETTINGS['account'][account]
    auth = None
    live = None
    # Get account
    if live_account is not None:
        live_account_settings = SETTINGS['live_account'][live_account]
        live = TqAccount(live_account_settings['broker_id'], live_account_settings['account_id'], live_account_settings['password'])
    else:
        auth = TqAuth(account_settings['username'], account_settings['password'])
    return auth, live


class Interval(NamedTuple):
    ONE_SEC: str = "1s"
    FIVE_SEC: str = "5s"
    ONE_MIN: str = "1m"
    FIVE_MIN: str = "5m"
    FIFTEEN_MIN: str = "15m"
    THIRTY_MIN: str = "30m"
    ONE_HOUR: str = "1h"
    FOUR_HOUR: str = "4h"
    ONE_DAY: str = "1d"
    TICK: str = "tick"

class InitOverallStep(NamedTuple):
    ONE_SEC: int = 2*60*60
    FIVE_SEC: int = 2*60*12
    ONE_MIN: int = 2*60

max_step_by_day = {
    "1s": 20700,
    "5s": 4140,
    "1m": 345,
    "5m": 72,
    "15m": 24,
}
