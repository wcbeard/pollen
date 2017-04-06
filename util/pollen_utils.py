import datetime as dt
import re

from bs4 import BeautifulSoup
from pandas import DataFrame, pandas as pd
from pandas.compat import lmap
import requests

udate_re = re.compile(r'.+?/(\d+)/(\d+)/(\d+)')


with open('../pfile.txt', 'r') as f:
    pollen_url = f.read().strip()


def count_url_2date(u):
    # print(u)
    m_ = udate_re.match(u)
    dates = m_.groups()
    return dt.date(*map(int, dates))


def parse_pollen_href(a) -> ('date', 'pollen_ct'):
    u = a.attrs['href']
    date = count_url_2date(u)
    if a.string and a.string.isdigit():
        ct = int(a.string)
    else:  # '', 'N/A'
        ct = -1
    return date, ct


def parse_pollen_page(html_txt) -> ('date', 'pollen_ct'):
    soup = BeautifulSoup(html_txt, "lxml")
    # sel = 'div.calendar-row.calendar-row-4 > div > div > span.count'
    sel = 'span.count > a'
    return DataFrame(lmap(parse_pollen_href, soup.select(sel)), columns=['Date', 'Count'])


def pollen_date2df(yr, m):
    u = pollen_url.format(year=yr, month=m)
    r = requests.get(u)
    return parse_pollen_page(r.content)


def pollen_data(yrmths):
    return pd.concat(
        [pollen_date2df(yr, m) for yr, m in yrmths],
        ignore_index=True
    ).sort_values('Date', ascending=True)


pscale = DataFrame([
    ['0', 'Absent'],
    ['1-14', 'Low'],
    ['15-89', 'Moderate'],
    ['90-1499', 'High'],
    ['1500+', 'Very High'],
], columns=['Count', 'Level'])
