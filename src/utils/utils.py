import yaml
import os.path

from tqsdk import TqApi, TqAuth, TqAccount

DIR = os.path.dirname(os.path.abspath(__file__))

def get_config() -> dict:
    # Read YAML file
    configFilePath = os.path.join(DIR, "/config.yaml")
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
    if live_account:
        live_account_settings = SETTINGS['live_account'][live_account]
        live = TqAccount(live_account_settings['broker_id'], live_account_settings['account_id'], live_account_settings['password'])
    else:
        auth = TqAuth(account_settings['username'], account_settings['password'])
    return auth, live
