import urllib.request
from bs4 import BeautifulSoup as bs
from pprint import pprint

import pandas as pd

def get_fee():
    url = "https://www.9qihuo.com/qihuoshouxufei?zhuli=true"
    html = urllib.request.urlopen(url).read()
    soup = bs(html, "html.parser")
    # find table with id heyuetbl
    table = soup.find("table", id="heyuetbl")
    # table to list
    table_list = table.find_all("tr")

    results = []
    prefix = "NaN"
    for row in table_list:
        tds = row.find_all("td")
        if len(tds) == 0:
            continue
        if len(tds) == 1:
            # section title 
            heading = tds[0].text
            if heading == "上海期货交易所":
                prefix = "SHFE"
            elif heading == "郑州商品交易所":
                prefix = "CZCE"
            elif heading == "大连商品交易所":
                prefix = "DCE"
            elif heading == "中国金融期货交易所":
                prefix = "CFFEX"
            elif heading == "上海国际能源交易中心":
                prefix = "INE"
        else:
            symbol = row.find("td", {"class": "heyuealink"})
            if symbol is not None:
                symbol = symbol.text
                name, symbol = symbol.split(" ")
                symbol = name + " " + prefix + "." + symbol.replace("(", "").replace(")", "")
            else:
                continue
            current_price = row.find("td", {"title": "当前价格（大概价格，不是实时的）"}).text
            today_limit = row.find("td", {"title": "今日涨/跌停板"}).text
            high_limit, low_limit = today_limit.split("/")
            margin_ratio = row.find("td", {"title": "多头保证金比例"}).text
            margin_ratio = margin_ratio.replace("%", "")
            margin = row.find("td", {"title": "每手保证金"}).text
            margin = margin.replace("元", "")
            open_fee = row.find("td", {"title": "开仓手续费(万分之几或者元)"}).text
            open_fee = open_fee.split("(")[-1].replace("元)", "")
            close_fee = row.find("td", {"title": "平昨仓手续费(万分之几或者元)"}).text
            close_fee = close_fee.split("(")[-1].replace("元)", "")
            close_today_fee = row.find("td", {"title": "平今仓手续费(万分之几或者元)"}).text
            close_today_fee = close_today_fee.split("(")[-1].replace("元)", "")
            gross_profit_per_tick = row.find("td", {"title": "每跳毛利"}).text
            total_fee = row.find("td", {"title": "一手合约手续费合计(开+平)"}).text
            total_fee = total_fee.replace("元", "")
            net_profit_per_tick = row.find("td", {"title": "浮盈一跳净盈利（每跳毛利-每跳手续费）"}).text

            net_profit_per_tick_per_k = float(net_profit_per_tick) / float(margin) * 1000

            result = {
                "symbol": symbol,
                "current_price": current_price,
                "high_limit": high_limit,
                "low_limit": low_limit,
                "margin_ratio": margin_ratio,
                "margin": margin,
                "open_fee": open_fee,
                "close_fee": close_fee,
                "close_today_fee": close_today_fee,
                "gross_profit_per_tick": gross_profit_per_tick,
                "total_fee": total_fee,
                "net_profit_per_tick": net_profit_per_tick,
                "net_profit_per_tick_per_k": net_profit_per_tick_per_k
            }
            results.append(result)
    
    # sort by net profit per tick per k
    results = sorted(results, key=lambda x: x["net_profit_per_tick_per_k"], reverse=True)
    return results

res = get_fee()
df = pd.DataFrame(res)
df.to_csv("fee.csv", index=False)