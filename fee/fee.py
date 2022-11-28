import urllib
from bs4 import BeautifulSoup as bs


def get_fee():
    url = "https://www.9qihuo.com/qihuoshouxufei?zhuli=true"
    html = urllib.request.urlopen(url).read()
    soup = bs(html, "html.parser")
    # find table with id heyuetbl
    table = soup.find("table", id="heyuetbl")
    # table to list
    table_list = table.find_all("tr")
    for row in table_list:
        tds = row.find_all("td")
        if len(tds) == 0:
            continue
        if len(tds) == 1:
            # section title 
            print(tds[0].text)
        else:
            symbol = row.find("td", {"class": "heyuealink"})
            if symbol is not None:
                symbol = symbol.text
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
            }
            print(result)
