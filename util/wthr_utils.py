from functools import partial
import numpy as np
from pandas import pandas as pd, DataFrame
from bs4 import BeautifulSoup
import requests

from util.utils import check_cached

wthr = ('https://www.wunderground.com/history/airport/KPDK/{yr}/{m}/1/'
        'MonthlyHistory.html?req_city=Alpharetta&req_state=GA&'
        'req_statename=Georgia&reqdb.zip=30022&reqdb.magic=1&reqdb.wmo=99999')


def clean_string_srs(s, nulls=['-']):
    """Some Series are strings bc of weird nullish values.
    This converts the srs to float
    """
    bm = s.isin(nulls)
    s2 = s.copy()
    s2.loc[bm] = np.nan
    return s2.astype(float)


def edit_html_table(tab: 'html') -> 'html':
    """Source has messed up table format;
    body in each row, 2 cols.
    """
    t = BeautifulSoup(str(tab), "lxml")
    orig_head = t.select_one('thead')
    new_head = t.select_one('tbody')
    orig_head.tr.replace_with(new_head.tr)

    for body in t.find_all('tbody'):
        body.replace_with_children()
    return t


def check_max_min(df, msg='Uh oh'):
    """Df columns has 2 levels; second
    level has 'Hi' and 'Lo'. This ensures
    that they are max/min of the level df.
    """
    df = df.assign(
        Max=lambda x: x.max(axis=1),
        Min=lambda x: x.min(axis=1),
    )  #
    samehi = df.Max == df.Hi
    assert samehi.all(), '{}\n\n{}'.format(msg, df[~samehi])
    samelo = df.Min == df.Lo
    assert samelo.all(), '{}\n\n{}'.format(msg, df[~samelo])
    # assert (df.Min == df.Lo).all(), msg


def check_max_min_df(df):
    for c in {c for c, _ in df}:
        # print(c)
        check_max_min(df[c], msg='Bad vals in {}: \n\n{}'.format(c, df[c]))


def process_wtrdf(df, fixer=None):
    lvl_1_cols = 'Temp Dew Humidity Press Vis Wind Prec'.split()
    lvl_2_cols = 'Hi Avg Lo'.split()
    cols = pd.MultiIndex.from_product([lvl_1_cols, lvl_2_cols])

    global wsub
    # Get subset of df w/ max/min/avg
    wsub = df.iloc[:, :18].copy()
    wsub.columns = cols[:18]

    # Fix null values
    str_cols = wsub.columns[wsub.dtypes == object]
    if len(str_cols):
        wsub[str_cols] = wsub[str_cols].apply(clean_string_srs)

    # Reorder cols for Wind
    wind = wsub['Wind'].copy()
    wind.columns = 'Avg Lo Hi'.split()
    wsub['Wind'] = wind

    # Fix and check data
    if fixer is not None:
        fixer(wsub)
    check_max_min_df(wsub.dropna(axis=0))

    # Add other cols
    wsub[('Prec', '')] = df['sum']
    wsub[('Event', '')] = df.iloc[:, -1]
    return wsub


def wtr_page2df(content: str, fixer=None) -> DataFrame:
    global df_orig
    soup = BeautifulSoup(content, "lxml")
    tab_orig = soup.select('table.daily')[0]
    tab = edit_html_table(tab_orig)
    df_orig = pd.read_html(str(tab), header=0, index_col=0)[0]
    df = process_wtrdf(df_orig.copy(), fixer=fixer)
    return df


def fix_df_exceptions(df, yr, m):
    if (yr, m) == (2013, 9):
        df.loc[17, ('Wind', 'Hi')] = np.nan
    elif (yr, m) == (2014, 7):
        df.loc[13, ('Wind', 'Hi')] = np.nan
    elif (yr, m) == (2014, 12):
        df.loc[2, ('Wind', 'Hi')] = np.nan
        df.loc[2, ('Wind', 'Avg')] = np.nan


def wtr_date2df(yr, month):
    u = wthr.format(yr=yr, m=month)
    r = requests.get(u)
    return wtr_page2df(r.content, fixer=partial(fix_df_exceptions, yr=yr, m=month))


@check_cached(.5)
def add_dates(df, yr=None, m=None):
    df.index.name = None
    return df.assign(Yr=yr, M=m).reset_index(drop=0).rename(columns={'index': 'Day'})


def wthr_data(yrmths, wtr_date2df=None):
    assert wtr_date2df
    wtr_dfs = [
        add_dates(wtr_date2df(yr, m), yr=yr, m=m)
        for yr, m in yrmths
    ]

    wtr_df = pd.concat(wtr_dfs, ignore_index=True)
    return wtr_df
