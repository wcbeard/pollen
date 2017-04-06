
# coding: utf-8

# In[ ]:

from imports import *


# In[ ]:

import re
import datetime as dt
import time
from collections import defaultdict
import json

# import requests
import requests_cache

from requests_cache import CachedSession
ss = CachedSession('pollen_darksky', backend='sqlite')

from bs4 import BeautifulSoup

# requests_cache.install_cache('pollen_darksky', backend='sqlite')
get_ipython().magic('matplotlib inline')

# %mkdir cache
import joblib; mem = joblib.Memory(cachedir='cache2')


# In[ ]:

from requests_cache.backends.redis import RedisDict
rd = RedisDict('red-cache')

import torch as T

get_ipython().magic('load_ext line_profiler')


# In[ ]:

import util.pollen_utils; reload(util.pollen_utils); from util.pollen_utils import (
    pollen_url, pollen_date2df as pollen_date2df_, pollen_data as pollen_data_
)

import util.utils; reload(util.utils); from util.utils import (
    check_one2one, yrmths, flatten_multindex, ends_with,
    batchify, test_batchify, collapse_dims, test_collapse_dims_,
    batch_getterer,
    to_sub_seqs, test_to_sub_seqs,
    ravel, repackage_hidden, mse
)

test_to_sub_seqs()
test_batchify()
pollen_date2df = mem.cache(pollen_date2df_)
pollen_data = mem.cache(pollen_data_)
test_collapse_dims_(T)
(";")


# In[ ]:

pd.options.display.max_columns = 40


# In[ ]:

# u = pollen_url.format(year=2014, month=1)


#     for yr in range(2000, 2018):
#         for m in range(1, 13):
#             u = url.format(year=yr, month=m)
#             r = requests.get(u)
#             print(yr, m, end='; ')

# # Pollen

# In[ ]:

# Sometimes Jan calendar will have counts from days
# at beginning of Feb or end or previous Dec.
# Just checking that they agree w/ numbers in
# those months' calendars before dropping dupe
# dates
poldf = pollen_data(yrmths)
check_one2one(poldf, 'Date', 'Count')

poldf = poldf.drop_duplicates('Date').reset_index(drop=1)
poldf.loc[poldf.Count == -1, 'Count'] = np.nan
poldf = poldf.dropna(axis=0)
poldf.Date = pd.to_datetime(poldf.Date)

poldf = poldf.assign(
    Yr =lambda x: x.Date.dt.year,
    M  =lambda x: x.Date.dt.month,
    D  =lambda x: x.Date.dt.day,
    Doy=lambda x: x.Date.dt.dayofyear,
)

poldf['Dayi'] = (poldf.Date - poldf.Date.min()).dt.days
poldf['Prev_cnt'] = poldf.Count.shift(1).fillna(method='ffill').fillna(0)  #.interpolate()
poldf['Prev_cnt_null'] = poldf.Dayi.shift() != (poldf.Dayi - 1)


# In[ ]:

from pandas.util.testing import assert_frame_equal


# In[ ]:

No = len(poldf2)


# In[ ]:

assert_frame_equal(poldf.reset_index(drop=1)[:No], poldf2.reset_index(drop=1))


# In[ ]:

feather.write_dataframe(poldf, 'cache/pollen.fth')


# In[ ]:

poldf2 = feather.read_dataframe('cache/pollen.fth')


#     cs = requests.session()
#     u = pollen_url.format(year=2017, month=4)
# 
#     cs.cache.delete_url(u)
#     r = requests.get(u)

#     !rm -r cache/joblib/util/pollen_utils/pollen_date2df/
#     !rm -r cache/joblib/util/pollen_utils/pollen_data/

#     for yr in range(2000, 2018):
#         for m in range(1, 13):
#             u = url.format(year=yr, month=m)
#             r = requests.get(u)
#             print(yr, m, end='; ')

# ## Process monthly calendar

# r = requests.get(u)
# soup = BeautifulSoup(r.content, "lxml")
# sel = 'div.calendar-row.calendar-row-4 > div > div > span.count > a'
# soup.select_one(sel)

# # Darksky

# In[ ]:

def mkds_url(sdate):
    loc = '33.7490,-84.3880'
    url_temp = 'https://api.darksky.net/forecast/{key}/{loc},{time}?exclude=flags'
    return url_temp.format(key=key, loc=loc, time=mktime(sdate))


with open('KEY.txt', 'r') as f:
    key = f.read().strip()


# #### Time series generator

# In[ ]:

def date_rng_gen(start):
    if isinstance(start, str):
        start = dt.datetime.strptime(start, '%Y-%m-%d')
    while 1:
        # print(start)
        rng = gen_yr_rng(start, backward=True)
        # print(rng[:5])
        start = rng[-1]
        for d in rng[:-1]:
            yield d

def gen_yr_rng(start, backward=True):
    start = start
    yr = start.year
    dprev = start.replace(year=yr - 1)
    rng = pd.date_range(start=dprev, end=start)
    if backward:
        return rng[::-1]
    return rng
    d2 = dt.datetime(*start.timetuple()[:-3])

def test_date_rng_gen():
    ts = list(it.islice(date_rng_gen('2017-03-01'), 400))
    a, b = ts[0], ts[-1]
    assert (pd.date_range(b, a)[::-1] == ts).all()
    
    s = pd.Series(ts)
    assert s.value_counts(normalize=0).max() == 1
    
    
def mktime(s, hour=12):
    # tm
    try:
        d = dt.datetime.strptime(s, "%Y-%d-%m")
    except TypeError:
        d = s
#     if isinstance(s, str):
#     else:
#         d = s
        
    if hour:
        d = d.replace(hour=hour)
    f = time.mktime(d.timetuple())
    return int(f)
    
test_date_rng_gen()


# ### Extract data

# In[ ]:

def camel2score(s):
    return s[0].upper() + ''.join([('_' + c.lower()) if c.isupper() else c for c in s[1:]])

def parse_data(j):
    hrs = j.pop('hourly')['data']
    # global hrdf
    js = json.dumps(hrs)
    hrdf = pd.read_json(js) #.rename(columns=camel2score)
    
    _dl = j.pop('daily')
    [[dl]] = _dl.values()
    cr = j.pop('currently')
    
#     if hrdf.shape != (24,15):
#         print(hrdf.shape)
#     assert hrdf.shape == (24,15), 'Hr shape: {}'.format(hrdf.shape)
    assert sorted(j) == ['latitude', 'longitude', 'offset', 'timezone']
    return hrdf, dl, cr, j

# hrdf, dl, cr, j = parse_data(r.json())


# In[ ]:

today = '2017-04-05'


# In[ ]:

# Pull new
for d in date_rng_gen(today):
    if d.month == d.day == 1:
        print(d.date())
    u = mkds_url(d)
    if (u in rd) or ss.cache.has_url(u):
        print('.', end='')
        continue
    r = ss.get(u)
    if r.status_code == 403:
        print('Forbidden')
        break
    if r.status_code != 200:
        print(d, r.status_code)


# In[ ]:

r = ss.get(u)


# In[ ]:

r.url


# ### Sync Sqlite db w/ Redis

# In[ ]:

def sync_rdd(ss, rd, stdate='2017-03-01'):
    n = 0
    for d in date_rng_gen(stdate):
        if d.month == d.day == 1:
            print(d.date(), end=' ')
        u = mkds_url(d)
        if u in rd:
            continue

        if not ss.cache.has_url(u):
            break
        r = ss.get(u)
        assert u == r.url
        rd[r.url] = r.json()
        print('.', end='')
        n += 1
    return n
        
sync_rdd(ss, rd, stdate=today)


# ### Accumulate and extract data

# In[ ]:

def accum(stdate='2017-03-01'):
    """Roll through the dates, pull out cached
    requests, and add parsed data to list"""
    dat = []

    for i, d in enumerate(date_rng_gen(stdate)):
#         if i > 20:
#             break
        if d.month == d.day == 1:
            print(d.date())
        u = mkds_url(d)
        # if ss.cache.has_url(u):
        #     r = ss.get(u)
        #     dat.append(parse_data(r.json()))
        if u in rd:
            j = rd[u]
            parsed = parse_data(j)
            dat.append(parsed)
            # print('.', end='')
        else:
            break
    return dat
    return


# %lprun -f pd.read_json accum(stdate='2017-03-01')

# In[ ]:

dat = accum(stdate='2017-04-04')


# In[ ]:

hrdfs, dls, crs, _ = zip(*dat)


# In[ ]:

dailydf = pd.read_json(json.dumps(dls)).rename(columns=camel2score)


# In[ ]:

def concat(dfs):
    all_cols = {c for df in dfs for c in df}
    col_dat = defaultdict(list)
    for cname in all_cols:
        for df in dfs:
            l = len(df)
            col_dat[cname].extend(df.get(cname, [None] * l))
    return DataFrame(col_dat)

get_ipython().magic('time hr_df = concat(hrdfs).rename(columns=camel2score)')


# In[ ]:

feather.write_dataframe(dailydf, 'cache/dark_day.fth')
# feather.write_dataframe(hr_df, 'cache/dark_hr.fth')


# In[ ]:

hr_dat[:2]


# ### Analysis of nulls

# In[ ]:

def rep_with_dummies_(df, col):
    df = df.copy()
    newcs = pd.get_dummies(df[col])
    for c in newcs:
        df[c] = newcs[c]
    return df.drop(col, axis=1)

def rep_with_dummies(df, cols):
    for c in cols:
        df = rep_with_dummies_(df, c)
    return df

hr_dat2 = rep_with_dummies(hr_dat, ['Icon', 'Summary', 'Precip_type'])


# In[ ]:

ns = (~(hr_dat == hr_dat))
ns.sum()


# In[ ]:

ncols = ns.sum()[ns.sum() > 0]


# In[ ]:

cn = hr_dat.eval('Cloud_cover != Cloud_cover').astype(int)


# In[ ]:




# In[ ]:

hr_dat2.corrwith(cn).sort_values(ascending=True)


# In[ ]:

hr_dat2[:3]


# In[ ]:




# In[ ]:







# In[ ]:




























