import modin.pandas as pd

def ema(series: pd.Series, n):
    """
    指数加权移动平均线: 求series序列n周期的指数加权移动平均

        计算公式:
            ema(x, n) = 2 * x / (n + 1) + (n - 1) * ema(x, n).shift(1) / (n + 1)

        注意:
            1. n 需大于等于1
            2. 对距离当前较近的k线赋予了较大的权重

    Args:
        series (pandas.Series): 数据序列

        n (int): 周期

    Returns:
        pandas.Series: 指数加权移动平均线序列

    Example::

        from tqsdk import TqApi, TqAuth, TqSim, tafunc

        api = TqApi(auth=TqAuth("信易账户", "账户密码"))
        klines = api.get_kline_serial("CFFEX.IF1908", 24 * 60 * 60)
        ema = tafunc.ema(klines.close, 5)
        print(list(ema))
    """
    ema_data = series.ewm(span=n, adjust=False).mean()
    return ema_data