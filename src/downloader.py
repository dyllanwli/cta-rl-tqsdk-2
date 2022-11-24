from datetime import date, datetime
from utils.utils import get_auth
from db.mongo import Mongo

def main():
    auth, _ = get_auth(account= "a1")
    dao = Mongo()
    # intervals = {'1s', '5s', '1m', '1d'}
    # symbol_list = ['soybean_oil']
    # dao.download_data(tqAPI.auth, symbol_list, date(2020, 9, 3), date(2022, 9, 1), intervals)
    intervals =  {'1s', '5s', '1m', '1d'}
    symbol_list = ['methanol']
    dao.download_data(auth, symbol_list, date(2016, 10, 6), date(2022, 11, 1), intervals)
    print(symbol_list, intervals, "downloaded")
if __name__ == "__main__":
    main()


