
# coding: utf-8

# In[ ]:

from imports import *


# In[ ]:

import re
import datetime as dt
import time

import requests
import requests_cache

from bs4 import BeautifulSoup

requests_cache.install_cache('pollen', backend='sqlite')
get_ipython().magic('matplotlib inline')

# %mkdir cache
import joblib; mem = joblib.Memory(cachedir='cache')


# In[ ]:

pd.options.display.max_columns = 30


# In[ ]:

import torch as T


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
date = lambda xs: dt.datetime(*xs)

test_to_sub_seqs()
test_batchify()
pollen_date2df = mem.cache(pollen_date2df_)
pollen_data = mem.cache(pollen_data_)
test_collapse_dims_(T)
(";")


# In[ ]:

# u = pollen_url.format(year=2014, month=1)


#     for yr in range(2000, 2018):
#         for m in range(1, 13):
#             u = url.format(year=yr, month=m)
#             r = requests.get(u)
#             print(yr, m, end='; ')

# ## Process monthly calendar

# body > div:nth-child(7) > div > div > div > div > div:nth-child(3) > div.calendar-wrapper > div > div.calendar-body.rows-5 > div.calendar-row.calendar-row-4 > div:nth-child(2) > div > span.count > a

# r = requests.get(u)
# soup = BeautifulSoup(r.content, "lxml")
# sel = 'div.calendar-row.calendar-row-4 > div > div > span.count > a'
# soup.select_one(sel)

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


# ### TODO: 
# - Historic average residuals
# - Compare embeddings for wthr w/ summaries
# - standardize time fields

# plt.figure(figsize=(16, 10))
# pdf.plot(y='Count', x='Date', ax=plt.gca())
# 
# d = pdf[:].reset_index(drop=0).set_index(['Doy', 'Yr']).Count.unstack()
# d
# 
# plt.figure(figsize=(16, 10))
# d.iloc[:150, 8:].plot(ax=plt.gca())

# # Get historical weather

#     # Src 1
#     import util.wthr_utils; reload(util.wthr_utils); from util.wthr_utils import (
#         wtr_date2df as wtr_date2df_, wthr_data as wthr_data_,
#         wthr, add_dates
#     )
# 
#     wtr_date2df = mem.cache(wtr_date2df_)
#     wthr_data = mem.cache(wthr_data_)

# In[ ]:

# Src 2
dailydf = feather.read_dataframe('cache/dark_day.fth')
hr_df = feather.read_dataframe('cache/dark_hr.fth')


# In[ ]:

dailydf['Dt'] = pd.to_datetime(dailydf.Time, unit='s')
dailydf['Day'] = dailydf.Dt.dt.day
dailydf['M'] = dailydf.Dt.dt.month
dailydf['Y'] = dailydf.Dt.dt.year


# In[ ]:

def rep_with_dummies_(df, col):
    df = df.copy()
    newcs = pd.get_dummies(df[col]).astype(int)
    for c in newcs:
        df[c] = newcs[c]
    return df.drop(col, axis=1)

def rep_with_dummies(df, cols):
    """Return a copy of df w/ each of `cols` replaced
    with its dummified self"""
    for c in cols:
        df = rep_with_dummies_(df, c)
    return df


# ### Float/nulls
# 
# #### Precip_type, Precip_accumulation

# In[ ]:

dailydf.loc[dailydf.eval('Precip_type != Precip_type'), 'Precip_type'] = 'none'
dailydf['Precip_accumulation'] = dailydf.Precip_accumulation.fillna(0)


# In[ ]:

dailydf['Min_time'] = dailydf.Dt.map(lambda t: int(t.replace(hour=0).strftime('%s')))


# In[ ]:

def fill_pimt_null(s, timecol):
    """This column is null when there is no precipitation.
    Not sure of anything better to do, so I'm just setting
    it to the minimum time of the day in question
    """
    s2 = s.copy()
    null_ptime = s.isnull()
    s2.loc[null_ptime] = timecol[null_ptime]
    return s2.astype(int)

dailydf.Precip_intensity_max_time = fill_pimt_null(dailydf.Precip_intensity_max_time, dailydf.Min_time)


# ### Cloud_cover
# - Only if snow
# Visibility
# Y Precip_intensity Humidity

# In[ ]:

ddf = rep_with_dummies(dailydf, 'Icon Precip_type'.split())  # Summary


# In[ ]:

from sklearn.ensemble import RandomForestRegressor

def fill_cloud_cover_null(cc, X):
    """Solution wasn't obvious, so I just imputed the nulls
    with a random forest using the other columns.
    """
    null = cc != cc

    rf = RandomForestRegressor(n_estimators=30, oob_score=True)
    rf.fit(X[~null], cc[~null])
    cc2 = cc.copy()
    cc2.loc[null] = rf.predict(X[null])
    
    return cc2

_feats = [k for k, v in ddf.dtypes.items() if (v == int) or (v == float) and k != 'Cloud_cover']
ddf['Cloud_cover'] = fill_cloud_cover_null(ddf.Cloud_cover, ddf[_feats])


# In[ ]:

assert (ddf == ddf).all().all()


# ### Check times

# In[ ]:

# Check that within a day the difference between maximum
# and minimum times are not greater than the
# number of seconds in a day

times = lfilter(lambda x: x.endswith('ime'), dailydf)
minmax = DataFrame({
    'Min': dailydf[times].min(axis=1),
    'Max': dailydf[times].max(axis=1),
}).assign(Diff=lambda x: x.Max.sub(x.Min).div(60 * 60 * 24)) 

assert 0 <= minmax.Diff.max() <= 1
minmax.Diff.max()  # should be no more than 1


# In[ ]:

assert (dailydf[times].min(axis=1) == dailydf.Min_time).all(), 'Min_time definition'


# In[ ]:

unix_time_to_day_hrs = lambda s, min_time: (s - min_time) / 3600

for t in set(times) - {'Min_time'}:
    c = t + 's'
    dailydf[c] = unix_time_to_day_hrs(dailydf[t], dailydf.Min_time)


# In[ ]:

from IPython.display import Image

Image('cloud_cover_model_perf.png', height=400, width=400)


# In[ ]:

hr_df.shape


# with requests_cache.disabled():
# #     r = requests.get(u)
#     wtr_date2df(2017, 3)

# In[ ]:

# wdf = wthr_data(yrmths)
wdf = wthr_data_(yrmths[:], wtr_date2df=wtr_date2df)

# Weird dupe row
wdf['Dt'] = lmap(lambda x: dt.datetime(*x), zip(wdf.Yr, wdf.M, wdf.Day))
_, bad_ix = wdf[wdf['Dt'] == "2001-03-31"].index
wdf = wdf[~(wdf.index == bad_ix)].copy().reset_index(drop=1)
assert wdf.Dt.value_counts(normalize=0).max() == 1

wdf['Doy'] = wdf.Dt.dt.dayofyear
# wdf = wdf.set_index('Dt')

# Fix 'T'
# Prec: '-': only in 2000; 'T': increases later.
bm = wdf['Prec'].isin(['T', '-'])
wdf.loc[bm, 'Prec'] = np.nan
wdf['Prec'] = wdf.Prec.astype(float)


# ### Look for Randomness in 'T'

#     wdf2 = wdf.assign(Tt=lambda x: ~x.Prec.isin(['T']), Dow=lambda x: x.Dt.dt.dayofweek)
#     wdf2.columns = flatten_multindex(wdf2)
# 
#     _dcs = 'Temp_hi Temp_lo Dew_hi Dew_lo Humidity_hi Humidity_lo Press_hi Press_lo Vis_hi Vis_lo'.split()
# 
#     vdf = wdf2.drop(['Day', 'Dt', 'Event'] + _dcs, axis=1).dropna(axis=0)
#     _cs = list(vdf)
#     vdf = vdf[_cs[-1:] + _cs[:-1]]
#     vdf = vdf.iloc[:, :8]
# 
#     Tix = vdf.query('~Tt').index.tolist()
#     Tnix = nr.choice(vdf.query('Tt').index, len(Tix)).tolist()
# 
#     sns.pairplot(data=vdf.ix[Tix + Tnix], hue="Tt",  markers="+",
#                       plot_kws=dict(s=50, edgecolor="b", linewidth=1, alpha=1))

# # Viz Wtr & Pollen

# In[ ]:

def log_(s):
    l = np.log10(s)
    bm = np.isnan(s) | np.isinf(s)
    l[np.isneginf(l)] = 0
#     l[np.isinf(l)] = l.max()
#     l[np.isnan(l)] = l.median()
    return l


# In[ ]:

xdf = wdf.assign(
    Cnt=lambda x: poldf.set_index('Date').Count.ix[x.Dt].tolist()
).assign(
    Logcnt=lambda x: log_(x.Cnt)
)
xdf.columns = flatten_multindex(xdf).assign(Prev_cnt=lambda x: x.Logcnt.shift())
# xdf = xdf.query('Doy <= 150')


# In[ ]:

s.shift(-1)


# In[ ]:

s = xdf.Logcnt[:5]
s


# In[ ]:

xdf.Event.value_counts(normalize=0)


# In[ ]:

xdf[:20]


# [To #Pytorch](#Pytorch)

# _cor = xdf.drop(['Dt', 'Cnt', 'Event', 'Vis_hi', 'Prec'], axis=1).dropna(axis=0).iloc[:, :].corr()
# sns.clustermap(_cor, standard_scale=1, annot=True, fmt='.1f')

# xdf.corr()

# plt.figure(figsize=(16, 10))
# gdf.Temp_lo.plot()
# gdf.Logcnt.div(gdf.Logcnt.max()).mul(90).plot()
# 

# ## Time Series model

# from sklearn.linear_model import LassoCV
# 
# la = LassoCV()

# In[ ]:

import pyflux as pf


# In[ ]:

his = ends_with('_hi')(xdf)
los = ends_with('_lo')(xdf)
xd = xdf.drop(his + los, axis=1).set_index('Dt')


# In[ ]:

xd[:2]


# ### ARIMAX

# In[ ]:

xdf[:2]


# In[ ]:

get_ipython().run_cell_magic('time', '', "model = pf.ARIMAX(data=xd.query('Yr == 2013'), formula='Logcnt ~ 1+ Temp_avg + Dew_avg + Press_avg', ar=1, ma=1, family=pf.Normal())\nm = model.fit()")


# In[ ]:

model.plot_fit(figsize=(15,10))


# In[ ]:

m.summary()


# In[ ]:

np.log(1)


# In[ ]:

model.plot_predict(h=150, past_values=100, oos_data=xd.query('Yr == 2014'))


# In[ ]:

model.plot_predict(h=150, past_values=100, oos_data=xd.query('Yr == 2014'))


# In[ ]:

xd.query('Yr == 2014').Logcnt.plot()


# In[ ]:

res = model.predict(h=150, oos_data=xd.query('Yr == 2014'), intervals=True)


# In[ ]:

res0 = model.predict(h=150, oos_data=xd.query('Yr == 2014'), intervals=False)


# In[ ]:

get_ipython().system('say done')


# In[ ]:

xd.groupby(['Yr', "M"]).size()


# In[ ]:

def pp(x, y=None, color=None):
    global x_, y_
    x_ = x
    y_ = y
    p = plt.plot(y)
    p = plt.plot(x)


# In[ ]:

y_


# In[ ]:

g = sns.FacetGrid(xd.set_index('Doy').query('Yr >= 2010 & Yr < 2017'), row='Yr', aspect=4)
# g.map(plt.plot, 'Logcnt')
g.map(pp, 'Logcnt', 'Prec')


# ## LSTM-Keras
# 
#     from keras.models import Sequential
#     from keras.layers.core import Dense, Activation, Dropout
#     from keras.layers.recurrent import LSTM
# 
#     xd['Nulls'] = xd.isnull().sum(axis=1)
#     X = xd.drop(['Event', 'Cnt',  'Logcnt'], axis=1).query('Nulls == 0')
# 
#     xd[:2]
# 
#     x13 = X.query('Yr == 2013').drop('Yr', axis=1)
#     y13 = xd.query('Yr == 2013 & Nulls == 0')[['Logcnt']]
# 
#     _, P = x13.shape
# 
#     x14 = X.query('Yr == 2014').drop('Yr', axis=1)
#     y14 = xd.query('Yr == 2014 & Nulls == 0')[['Logcnt']]
# 
#     x13.shape, y13.shape, P

# ## Pytorch

# In[ ]:

import toolz.curried as z

import torch as T
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from functools import wraps


# In[ ]:

from collections import defaultdict
from sklearn.preprocessing import StandardScaler

from pandas.compat import lrange

tofloat = lambda x: x.data[0]


# In[ ]:

ycol = 'Logcnt'
feats = 'Temp_avg Dew_avg Humidity_avg Wind_avg Prec M Doy'.split()
rx_ = xdf[feats + [ycol]].copy()

# Null handling
rx_['Prec'] = rx_['Prec'].fillna(0)
rx_ = rx_.dropna(axis=0)
ry = rx_.pop(ycol).astype(dtype=np.float32)

# sss = defaultdict(lambda: StandardScaler())
ss = StandardScaler()
rx = ss.fit_transform(rx_).astype(dtype=np.float32)

rx.shape


# nr.seed(0)
# P = 3
# N = 1000
#     # x_, y_ = gen_dat2(P=P, N=N, dtype=np.float32)

# nhidden = 21
# num_layers = 1
# seq_len = 50
# bcz = 10
# 
# to_sub = lambda x: T.from_numpy(to_sub_seqs(x, seq_len=seq_len))
# xt = to_sub(rx)
# yt = to_sub(ry)
# 
# batch_getter, brange = batch_getterer(xt, y=yt, batch_size=bcz, var=True)
# print('brange', brange)
# h0 = Variable(T.randn(num_layers, bcz, nhidden))
# 
# 
# # Write relevant y elements
# L = np.sum([np.prod(batch_getter(b)[1].size()[:2]) for b in brange])
# _yeval = yt.numpy().ravel()[:L]
# feather.write_dataframe(DataFrame(_yeval), '/tmp/res/y.fth')
# 
# xt.size()

# In[ ]:

nhidden = 21
num_layers = 1
seq_len = 50
bcz = 10

xt = T.from_numpy(rx)
yt = T.from_numpy(ry.values[:, None])

print(yt.size())
print(xt.size())

criterion = nn.MSELoss()


# In[ ]:

def get_batch(source, i, bptt=20, evaluation=False):
    seq_len = min(bptt, len(source) - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    return data

def seq_batch_iter(x, y=None, bptt=20, evaluation=False):
    """Iterates according to schema in
    http://stackoverflow.com/a/37009670/386279
    Every transition seen.
    """
    for i in range(len(x) - 1):
        xb = get_batch(x, i, bptt=bptt, evaluation=evaluation)
        if y is None:
            yield xb
        else:
            yield xb, get_batch(y, i, bptt=bptt, evaluation=evaluation)
            
def batch_iter(x, y=None, bptt=20, evaluation=False):
    for i in range(0, len(x) - 1, bptt):
        xb = get_batch(x, i, bptt=bptt, evaluation=evaluation)
        if y is None:
            yield xb
        else:
            yield xb, get_batch(y, i, bptt=bptt, evaluation=evaluation)


# In[ ]:

class Rnn(nn.Module):
    def __init__(self, P=3, nhidden=21, num_layers=2):
        super().__init__()
        self.P, self.nhidden, self.num_layers = (
            P, nhidden, num_layers
        )
        self.rnn = nn.GRU(P, nhidden, num_layers, batch_first=True)
        # self.rnn.zero_grad()
        
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()
        self.zero_grad()

    def __dir__(self):
        return super().__dir__() + list(self._modules)
    
    def forward(self, input, hidden=None, outputh=False, update_hidden=True):
        if hidden is None:
            hidden = self.hidden
        out1, hout = self.rnn(input, hidden)
        out2 = self.decoder(ravel(out1))
        self.hidden = repackage_hidden(hout)
        if outputh:
            return out2, hout
        return out2
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        "For lstm I'll need to return 2"
        weight = next(self.rnn.parameters()).data
        mkvar = lambda: Variable(weight.new(self.num_layers, bsz, self.nhidden).zero_())
        return mkvar()
    
    def set_hidden(self, bsz):
        h = self.init_hidden(bsz)
        self.hidden = h
    

m = model = Rnn(P=rx.shape[-1], nhidden=nhidden, num_layers=num_layers)
model.set_hidden(bcz)
# m = model = Rnn()

# optimizer = optim.Adam(model.parameters(), lr = 0.005)
optimizer = optim.RMSprop(model.parameters(), lr = 0.005)
model


# In[ ]:

class BatchArraySingle(object):
    def __init__(self, x=None, seq_len=5, truncate=True, tovar=True):
        self.truncate = truncate
        self.x = x
        self.N = len(x)
        self.seq_len = seq_len
        
        if truncate:
            self.rem = BatchArraySingle(x=x, seq_len=seq_len, truncate=False)
            n_segs = self.N // seq_len
            # print(n_segs)
            self.ix = np.arange(0, n_segs * seq_len, seq_len, dtype=int)
            retfuncs = [Variable, T.stack] if tovar else [T.stack]
        else:
            # remainder
            self.ix = np.arange(0, self.N, seq_len, dtype=int)
            retfuncs = [list, z.map(Variable)] if tovar else []

        self.retfunc = z.compose(*retfuncs)
    
    def __getitem__(self, ix):
        bixs = self.ix[ix]
        if isint(bixs):
            batches = self.x[bixs:bixs+self.seq_len]
        else:
            batches = [self.x[bix:bix+self.seq_len] for bix in bixs]
        return self.retfunc(batches)

    idxmax = property(lambda x: len(x.ix) - 1)
    
    
class BatchArray(BatchArraySingle):
    def __init__(self, x=None, y=None, seq_len=5, truncate=True, tovar=True, batch_size=20):
        super().__init__(x=x, seq_len=seq_len, truncate=truncate, tovar=tovar)
        
        self.xb = BatchArraySingle(x=x, seq_len=seq_len, truncate=truncate, tovar=tovar)
        if truncate:
            self.rem = BatchArray(x=x, y=y, seq_len=seq_len, truncate=False, tovar=tovar)
        
        self.y = y
        self.yb = BatchArraySingle(x=y, seq_len=seq_len, truncate=truncate, tovar=tovar)
        self.batch_size = batch_size

    def __getitem__(self, ix):
        return self.xb.__getitem__(ix), self.yb.__getitem__(ix)
    
    def batch_ix_iter(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        nb = len(self.ix) // batch_size
        return np.arange(nb * batch_size).reshape(-1, batch_size)

isint = lambda x: np.issubdtype(type(x), int)

# ba = BatchArray(x=x[:], y=y[:], seq_len=2)


# In[ ]:

def train_epoch(barray, model=None, hidden=None, optimizer=None, eval=False, batch_size=None):
    batch_size = batch_size or barray.batch_size
    assert batch_size or hidden
    if hidden is None:
        hidden = model.init_hidden(batch_size)
    tot_loss = 0
    res = []
    global x, y, all_elemsb
#     all_elemsb = set()

    # for x, y in batch_iter(xt, y=yt, bptt=bptt, evaluation=eval):
    for bix in barray.batch_ix_iter(batch_size=batch_size):
        x, y = barray[bix]
#         for batch in x.data:
#             for row in batch:
#                 all_elemsb = all_elemsb | set([tuple(row.numpy().ravel())])
        optimizer.zero_grad()
        # output, hidden = model(x, hidden)
        output = model(x, hidden)
        # hidden = repackage_hidden(hidden)
        
        res.append(output.data.squeeze())
        if eval:
            continue

        loss = criterion(output, y.view(-1, 1))
        loss.backward()

        T.nn.utils.clip_grad_norm(model.parameters(), 3)

    #     norms = []
        maxnorm = max(T.norm(p.grad.data) for p in m.parameters())
        if maxnorm > train_epoch.mnorm:
            train_epoch.mnorm = maxnorm
            print('max(grad) = {:.3f}'.format(maxnorm))

        optimizer.step()
        tot_loss += loss
    res = T.stack(res).view(-1).numpy()
    if eval:
        return res
    return tot_loss, res
     
train_epoch.mnorm = 0
# tot_loss, hidden, res = train(model=model, hidden=None, brange=brange, batch_getter=batch_getter, optimizer=optimizer)
# print(tofloat(tot_loss))


# In[ ]:

model.set_hidden(10)
ba = BatchArray(x=xt, y=yt, seq_len=50, batch_size=10)
trn_y = np.concatenate([ba[bix][1].data.numpy().ravel() for bix in ba.batch_ix_iter()])
feather.write_dataframe(DataFrame(trn_y), '/tmp/res/y.fth')
L = len(trn_y)
# ba.batch_ix_iter()


# In[ ]:

get_ipython().magic('mkdir /tmp/res')


# In[ ]:

mse(trn_y, res)


# In[ ]:

for bix in ba.batch_ix_iter():
    break


# In[ ]:

tot_loss, res = train_epoch(ba, model=model, hidden=None, optimizer=optimizer)
# tot_loss, res = train_epoch(xt, yt, model=model, hidden=None, bptt=50, optimizer=optimizer)
# tot_loss, res = train(model=model, hidden=None, brange=brange, batch_getter=batch_getter, optimizer=optimizer)
tot_loss


# In[ ]:

get_ipython().system('mv /tmp/res/ /tmp/res_10/')
get_ipython().magic('mkdir /tmp/res')


# In[ ]:

get_ipython().system('rm /tmp/res/x*')


# In[ ]:

print('Time: {:.2f}'.format(tt))
print('Acc: {:.2f}'.format(mse(res, trn_y)))


# In[ ]:

st = time.perf_counter()
# losses = []
# losses.append((mse(_yeval, res), tofloat(tot_loss)))

# for i in range(1):
for i in range(2000):
# for i in range(1000, 2000):
# for i in range(2000, 4000):
# for i in range(4000, 10000):
    tot_loss, res = train_epoch(ba, model=model, hidden=None, optimizer=optimizer)

    print('.', end='')
    if i % 99 == 0:
        print()
        print('{:,.3f}; {:,.3f}'.format(mse(trn_y, res), tofloat(tot_loss)))
        feather.write_dataframe(DataFrame(res), '/tmp/res/x{:05}.fth'.format(i))

tt = time.perf_counter() - st


# In[ ]:

res = train_epoch(ba, model=model, optimizer=optimizer, eval=True)


# In[ ]:

print('Time: {:.2f}'.format(tt))
print('Acc: {:.2f}'.format(mse(res, trn_y)))


# In[ ]:

PATH = 'model/gru3.mod'


# In[ ]:

T.save(model.state_dict(), PATH)


# In[ ]:

# T.load(model.state_dict(), 'model/gru1.mod')
model.load_state_dict(T.load(PATH))


# ### Eval unseen

# In[ ]:

leftovers = np.prod(yt.size()) - L
batches_left = leftovers / seq_len
print('leftovers:', leftovers)
print('batches_left:', batches_left)


# In[ ]:

xdf[:2]


# In[ ]:

xt


# In[ ]:

xtst = collapse_dims(xt, [(0, 1), (2,)])[-leftovers:]
# xtst = 
ytst = collapse_dims(yt, [(0, 1)])[-leftovers:]
# xtst, ytst = map(Variable, [xtst, ytst])


# ## Plot comparisons

# In[ ]:

def date_range_df(d1, d2):
    if isinstance(d1, int):
        rg1 = pd.date_range('2017-01-01', '2017-12-31', freq='d')
        d1, d2 = rg1[d1], rg1[d2]
    drg = pd.date_range(d1, d2, freq='MS')

    return drg
date_range_df(1, 200)


# In[ ]:

len(xdf)


# In[ ]:

res = train_epoch(ba, model=model, optimizer=optimizer, eval=True)


# In[ ]:

pdf_ = xdf.ix[rx_.index].assign(D=lambda x: x.Dt.dt.day)
pdf_['Pred'] = 0
pdf_.loc[pdf_.index[:L], 'Pred'] = res

pdf = pdf_.query('Yr >= 2013 & M < 6')
this_year = pdf.query('Yr == 2017')
# 'Doy', 'Temp_avg', 'M', 'Dt'

def plot(x, y, m, date, dashed=None, **_):
    global d
    ycol = y.name
    d = date

    bm = m <= 4
    if dashed is None:
        plt.plot(x, y)
    else:
        plt.plot(x, y, '--')
    
    drg = pd.date_range(date.iloc[0], date.iloc[-1], freq='MS')
    yo, yi = plt.ylim()
    xd = drg.dayofyear
    labs = drg.strftime("%b")
    plt.vlines(xd, 0, yi / 10)
    
    for xl, lab in zip(xd, labs):
        plt.text(xl, yi / 7, lab)
        
    if dashed is None:
        plt.plot(this_year.Doy, this_year[ycol], '--')
#         plot(this_year.Doy, this_year.Temp_avg, this_year.M, this_year.Dt, dashed=True, **_)
    else:
        return
#     plt.legend()


# In[ ]:

pdf[:1]


# In[ ]:

len(rx)


# In[ ]:

len(pdf_)


# In[ ]:

L


# In[ ]:

res


# In[ ]:




# In[ ]:

# Check DF alignment by making sure they're 100% correlated
# after StandardScaler
fcols = pdf_.columns[(pdf_.dtypes == float) | (pdf_.dtypes == int)]
_, cols_ = xt.size()
for col in range(cols_):
    feat = xt.numpy()[:, col]
    cor = pdf_[fcols].reset_index(drop=1).corrwith(Series(feat)).round(4)
    close_col = cor.idxmax()
    assert cor[close_col] == 1
    print('{}: {:.2%}'.format(close_col, cor[close_col]))
#     break


# In[ ]:

from pandas import Series


# In[ ]:

g = sns.FacetGrid(data=pdf, row='Yr', aspect=6)
# g.map(plot, 'Doy', 'Cnt', 'M', 'Dt')
g.map(plot, 'Doy', 'Pred', 'M', 'Dt', dashed=True)
g.map(plot, 'Doy', 'Logcnt', 'M', 'Dt', dashed=True)
plt.legend(['Predicted', 'Actual'])
# plt.savefig('plots/annual_temps.png', bbox_inches='tight', dpi=200)


# ### Held out

# In[ ]:

l = len(xt) - L


# In[ ]:

ldays = rx_.iloc[-2*l:].assign(Yr=lambda x: (x.Doy.shift() > x.Doy).fillna(False).astype(int).cumsum().add(2013))
ldays['Dt'] = np.concatenate([pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')[gdf.Doy - 1].date
                for yr, gdf in ldays.groupby(['Yr'], sort=False)])

ld1, ld2 = np.array_split(ldays, 2)
# yr2015 = pd.date_range('2015-01-01', '2016-01-01', freq='D')[ld1 - 1]
# yr2016 = pd.date_range('2016-01-01', '2017-01-01')[ld2 - 1]


# In[ ]:

unvar = lambda x: x.data.numpy().ravel()


# In[ ]:

# Create data
ba2 = BatchArray(x=xt[-2 * l:], y=yt[-2 * l:], seq_len=l, batch_size=1)
xs_, ysv_ = ba2[[0]]
xs, ysv = ba2[[1]]

ys_ = Series(unvar(ysv_), index=ld1.Dt)
ys = Series(unvar(ysv), index=ld2.Dt)


# In[ ]:

# Run prediction
model.set_hidden(1)
yspred_ = model(xs_, update_hidden=True)
yspred = model(xs, update_hidden=True)

yspred_ = Series(unvar(yspred_), index=ld1.Dt)
yspred = Series(unvar(yspred), index=ld2.Dt)


# In[ ]:

_, [ax1, ax2] = plt.subplots(2, 1, figsize=(20, 20))

yspred_.plot(style='.-', ax=ax1, label='Pred')
ys_.plot(style='.-', ax=ax1, label='Y')

yspred.plot(style='.-', ax=ax2, label='Pred')
ys.plot(style='.-', ax=ax2, label='Y')

plt.legend()


# In[ ]:

n = -90
plt.plot(yr2015.date[:n], _y[:n], '.-')


# In[ ]:

_y = unvar(yspred_)
plt.plot(yr2015, _y, '.-')


# In[ ]:

_, [ax1, ax2] = plt.subplots(2, 1, figsize=(20, 20))
ax1.plot(unvar(yspred_), '.-')
ax1.plot(unvar(ys_), '.-')
# ax1.set_xticklabels(ld1)
ax1.set_xticklabels(yr2015)
ax1.legend(['Pred', 'Cnt'])

ax2.plot(unvar(yspred), '.-')
ax2.plot(unvar(ys), '.-')
ax2.legend(['Pred', 'Cnt'])


# In[ ]:

plt.figure(figsize=(16, 10))
plt.plot(unvar(yspred))
plt.plot(unvar(ys))
plt.legend(['Pred', 'Cnt'])


# In[ ]:

model.set_hidden(10)
ba = BatchArray(x=xt, y=yt, seq_len=50, batch_size=10)
trn_y = np.concatenate([ba[bix][1].data.numpy().ravel() for bix in ba.batch_ix_iter()])
feather.write_dataframe(DataFrame(trn_y), '/tmp/res/y.fth')
L = len(trn_y)
# ba.batch_ix_iter()


# In[ ]:

model.forward()


# In[ ]:

xs


# In[ ]:

resho = train_epoch(ba2, model=model, optimizer=optimizer, eval=True)


# In[ ]:

trn_y = np.concatenate([ba[bix][1].data.numpy().ravel() for bix in ba.batch_ix_iter()])


# In[ ]:

plt.figure(figsize=(16, 10))
ax = plt.gca()
pdf.query('Yr >= 2012').set_index(['Doy', 'Yr']).Temp_avg.unstack().plot(ax=ax)  #.fillna(0)


# In[ ]:

pdf[:2]


# In[ ]:




# In[ ]:

plt.figure(figsize=(20, 10))
xdf.query('Yr > 2014').set_index('Dt').Cnt.plot()


# In[ ]:

batches_left


# In[ ]:

seq_len


# In[ ]:

87 * 50


# In[ ]:

xt


# In[ ]:

_x, _y = batch_getter(0)


# In[ ]:

f = float(np.array(1))
f


# In[ ]:

type(f)


# In[ ]:

float(np.array([1]))


# In[ ]:

tot_loss.


# In[ ]:

tot_loss


# In[ ]:

float(tot_loss)


# In[ ]:


_x, _y = batch_getter(0)

optimizer.zero_grad()
output, h1 = model(_x, h0)
h1 = repackage_hidden(h1)

loss = criterion(output, _y.view(-1, 1))
loss.backward()

optimizer.step()

loss


# In[ ]:

optimizer.zero_grad()
output2, h2 = model(_x, h2)
h2 = repackage_hidden(h2)

loss = criterion(output2, _y.view(-1, 1))
loss.backward()

optimizer.step()

loss


# In[ ]:

output2


# In[ ]:

_y.view(-1, 1)


# In[ ]:

m.decoder.bias


# In[ ]:

output


# In[ ]:

_y.view(-1, 1)


# In[ ]:

_y


# In[ ]:

_x


# In[ ]:

_y


# In[ ]:

_x


# In[ ]:

output


# In[ ]:

rnn = nn.GRU(P, nhidden, num_layers, batch_first=True)
rnn.zero_grad()
decoder = nn.Linear(nhidden, 1)

input = Variable(T.randn(bcz, seq_len, P))
h0 = Variable(T.randn(num_layers, bcz, nhidden))

# output, hn = rnn(input, h0)


# In[ ]:

_x, _y = batch_getter(0)
output, h1 = rnn(_x, h0)


# In[ ]:

out2 = decoder(output.squeeze())


# In[ ]:

decoder


# In[ ]:

out2


# In[ ]:

input.data.size()


# In[ ]:

output.data.size()


# In[ ]:

output.size()


# In[ ]:

h1.data.size()


#     # rnn = nn.LSTM(P, nhidden, num_layers)
#     input = Variable(T.randn(5, bcz, P))
#     h0 = Variable(T.randn(bcz, num_layers, nhidden))
#     # c0 = Variable(T.randn(num_layers, bcz, nhidden))
#     output, hn = rnn(input, h0)

# In[ ]:

rnn = nn.LSTM(P, nhidden, num_layers)
input = Variable(torch.randn(5, bcz, P))
h0 = Variable(torch.randn(num_layers, bcz, nhidden))
c0 = Variable(torch.randn(num_layers, bcz, nhidden))
output, hn = rnn(input, (h0, c0))


# In[ ]:

rnn(b1)


# In[ ]:





np.random.seed(0)
x = np.random.randint(-5, 5, size=(N, P)).astype(np.float32)
xsub = to_sub_seqs(x, seq_len=3)
xt = torch.from_numpy(xsub)

rnn = nn.GRU(P, nhidden, 1, batch_first=True)
rnn.zero_grad()


# In[ ]:

x2[0]


# In[ ]:

xx[0]


# In[ ]:

r(v, hidden)


# In[ ]:

xx = 


# In[ ]:

# x0 = T.unsqueeze(xx[0], 0)
x0 = xx[0:4]
v = Variable(x0, requires_grad=True)
rnn(v)


# In[ ]:

lin = nn.Linear(3, 20)
lin


# In[ ]:

lin(Variable(xx))


# In[ ]:

T.randn(1, 3)


# In[ ]:

sm[:10]


# In[ ]:

np.r_[0, sm[:-1]][:10]


# In[ ]:

_t[:10]


# In[ ]:

_d.sum(dim=1)[:10]


# class Rnn(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
#         self.fc2   = nn.Linear(120, 84)
#         self.fc3   = nn.Linear(84, 10)
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#     def num_flat_features(self, x):
#         size = x.size()[1:] # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#     
# Rnn()

# In[ ]:




# In[ ]:

xx


# In[ ]:

yy


# ## Chainer

# In[ ]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable, Reporter
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


# #### RNN
# Dummy data set

# In[ ]:

df2var = lambda x: Variable(x.values.astype('f'))
tovar = lambda x: Variable(x.astype('f'))
Variable.flt = lambda x: float(x.data)


# In[ ]:

def batch_train_(model, x, y, batch_size=50, clip=None):
    rnn = model.predictor
    loss = 0
    N = len(y)
    ix_rng = range(0, N, batch_size)
    final = len(ix_rng) - 1
    for i, ix in enumerate(ix_rng):
        loss += model(x[ix:ix+batch_size], y[ix:ix+batch_size])
        # print('{}-{}'.format(ix, ix+batch_size), end=';')
        if ((i + 1) % 10 == 0) or (i == final):
            # print('.!.', end='')
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            if clip:
                optimizer.clip_grads(clip)
            optimizer.update()
            rnn.reset_state()  # ?
            loss = 0
    return model


# In[ ]:

train_len = 3
batch_len = 2

for b in range(batch_len):
    break


# In[ ]:

_x, *_ = x
_x


# In[ ]:

r1, r2 = batch


# In[ ]:




# In[ ]:

class Regressor(Chain):
    def __init__(self, predictor):
        super().__init__(predictor=predictor)
        
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        reporter.report({'loss': loss}, self)
        return loss
    

class RNN(Chain):
    def __init__(self, P):
        super().__init__(
            in_=L.Linear(P, 100),  # word embedding
            mid=L.LSTM(100, 128),  # the first LSTM layer
            out=L.Linear(128, 1),  # the feed-forward output layer
        )

    def reset_state(self):
        # print('RS!')
        self.mid.reset_state()

    def __call__(self, X):
        # Given the current data, predict the next data point.
        return z.pipe(X, self.in_, self.mid, self.out)


# In[ ]:

def mk_model(P):
    rnn = RNN(P)             # x    -> pred
    model = Regressor(rnn)  # x, t -> score
    model.cleargrads()

    optimizer = optimizers.Adam()
    # optimizer = optimizers.SGD()
    optimizer.use_cleargrads()
    optimizer.setup(model)

    reporter = Reporter()
    reporter.add_observer('model', model)
    return model, rnn, reporter, optimizer


# ### Fake data

# In[ ]:




# # TODO: principled approach to iters, batch size, epochs

# In[ ]:

(np.prod([8,8,3]) + 1) * np.prod([14,14,20])


# In[ ]:

input_height, filter_height = 32, 8
P = 1
S = 2
(input_height - filter_height + 2 * P)/S + 1


# In[ ]:

model, rnn, reporter, optimizer = mk_model(3)


# In[ ]:

from itertools import count


# In[ ]:

def batch_train(model, x, y, batch_size=5, train_len=100, clip=None):
    mk_batch = partial(batchify, batch_size=batch_size, train_len=train_len)
    rnn = model.predictor
    loss = 0
    N = len(y)
    
    # TODO: simplify
    ix_rng = range(0, N, batch_size)
    final = len(ix_rng) - 1
    
#     TODO: why no len???
#     mb = mk_batch(x)
    print(len(mk_batch(y)))
    print(len(mk_batch(x)))
    
    for i, batchx, batchy in zip(
        count(), mk_batch(x), mk_batch(y)):
        print(i)
        loss += model(tovar(batchx), tovar(batchy))
        # print('{}-{}'.format(ix, ix+batch_size), end=';')
        if ((i + 1) % 10 == 0) or (i == final):
            print('.!.', end='')
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            if clip:
                optimizer.clip_grads(clip)
            optimizer.update()
            rnn.reset_state()  # ?
            loss = 0
    return model


# In[ ]:

model.predictor.in_.W.shape


# In[ ]:

InvalidType: 
Invalid operation is performed in: LinearFunction (Forward)

Expect: prod(in_types[0].shape[1:]) == in_types[1].shape[1]
Actual: 300 != 3


# In[ ]:

for _ in range(1):
    batch_train(model, xx.data, yy.data, batch_size=5, train_len=100)
rnn.reset_state()
# model(xx.data, yy.data).flt()


# In[ ]:

mk_batch = partial(batchify, batch_size=2, train_len=100)


# In[ ]:

bs = mk_batch(xx.data)


# In[ ]:

len(bs)


# In[ ]:

bs


# In[ ]:

batchify(xx.data, 5, 100)


# In[ ]:

len(xx)


# In[ ]:

len(yy)


# In[ ]:

yy.data


# In[ ]:

def compute_loss_(x, y, batch_size=5):
    loss = 0
    for i in range(0, len(y) - batch_size, batch_size):
        loss += model(x[i:i+batch_size], y[i:i+batch_size])
    return loss


# In[ ]:

def compute_loss(x, y):
    loss = 0
    for i in range(0, len(y)):
        loss += model(x[i:i+1], y[i:i+1])
        print(loss.flt())
    return loss


#     # All at once ok w/ N=1000
#     for i in range(20):
#         loss = model(xx, yy)
# 
#         model.cleargrads()
#         loss.backward()
#     #     loss.unchain_backward()
#      #   optimizer.clip_grads(5)
#         optimizer.update()
# 
#     
#     
#     # All at once ok w/ N=1000
#     for i in range(20):
#         loss = model(xx, yy)
# 
#         if i:
#             model.cleargrads()
#             loss.backward()
#             optimizer.clip_grads(50)
#             optimizer.update()

# In[ ]:




# In[ ]:




# In[ ]:

rnn = RNN()             # x    -> pred
model = Regressor(rnn)  # x, t -> score
model.cleargrads()

optimizer = optimizers.Adam()
# optimizer = optimizers.SGD()
optimizer.use_cleargrads()
optimizer.setup(model)

reporter = Reporter()
reporter.add_observer('model', model)


# In[ ]:

# All at once ok w/ N=1000
_rng = np.arange(N)
for i in range(20):
    loss = model(xx, yy)
    
    # if (i + 1) % 50 == 0:
    if i:
        model.cleargrads()
        loss.backward()
    #     loss.unchain_backward()
        optimizer.clip_grads(50)
        optimizer.update()

loss.flt()


# In[ ]:

_rng = np.arange(N)


# In[ ]:

n = 20
r = range(n)
b = 3
jump = n // b
list(range(0, jump))


# In[ ]:

for i in list(range(0, jump)):
    print(i)
    ix = [r[(jump * j + i) % n] for j in range(b)]
    print(ix)


# In[ ]:




# In[ ]:

len(yy)


# In[ ]:

len(ix_rng)


# In[ ]:




# In[ ]:

# Batch size five

loss = 0
BATCH_SIZE = 50
ix_rng = range(0, N, BATCH_SIZE)
# for i in range(100):
for i, ix in enumerate(ix_rng):
    # loss = compute_loss(xn, yn)
    loss += model(xx[ix:ix+BATCH_SIZE], yy[ix:ix+BATCH_SIZE])
    print('{}-{}'.format(ix, ix+BATCH_SIZE), end=';')
    if ((i + 1) % 10 == 0) or (i == len(ix_rng) - 1):
        print('.!.', end='')
        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
#         optimizer.clip_grads(500)
        optimizer.update()
        rnn.reset_state()  # ?
        loss = 0

# loss.flt()

rnn.reset_state()
model(xx, yy).flt()


# In[ ]:

get_ipython().set_next_input('x.shape[0] -> batch size');get_ipython().magic('pinfo size')


# In[ ]:

rnn.reset_state()

plt.plot(yy.data.ravel(), rnn(xx).data.ravel(), '.')


# In[ ]:

DataFrame(OrderedDict([('Act', yy.data.ravel()), ('Pred', rnn(xx).data.ravel().round(1))]))


# In[ ]:

rnn.mid.h.shape


# In[ ]:

xx.size


# In[ ]:

xx.shape


# In[ ]:

yy.data.ravel()[:20]


# In[ ]:




# In[ ]:

compute_loss(xn, yn)


# In[ ]:

BATCH_SIZE = 5
loss = 0
rnn.reset_state()
model.cleargrads()
for i in range(0, len(yn), BATCH_SIZE):
    # print(i, i + BATCH_SIZE)
    loss += model(xn[i:i+BATCH_SIZE], yn[i:i+BATCH_SIZE])
    print(float(loss.data))


# In[ ]:




# In[ ]:

loss.data


# In[ ]:

for x, y in zip(xn, yn):
    model(x, y)
    break


# In[ ]:

xn.data.ndim


# In[ ]:

x.data.ndim


# In[ ]:

y.data.ndim


# In[ ]:

def compute_loss(xs, ys):
    loss = 0
    for x, y in zip(xs, ys):
        loss += model(x, y)
    return loss


# In[ ]:

compute_loss(xn, yn)


# In[ ]:

x.data


# In[ ]:

loss = model(xn, yn)
show_grads()
loss.backward()
show_grads()
optimizer.update()
show_grads()

loss.data


# In[ ]:

valloss = model(xs, ys)
model.cleargrads()
valloss.data


# In[ ]:

pred = rnn(xn)
pred.data


# In[ ]:

y14


# In[ ]:

reporter.observation['model/loss'].data


# In[ ]:

((y13 - pred.data) ** 2).sum() / len(y13)


# In[ ]:

pred.data


# In[ ]:

xn.volatile


# In[ ]:

show_grads = lambda: print(rnn.in_.W.grad[:2, :5])


# In[ ]:




# In[ ]:




# In[ ]:

print(rnn.in_.W.grad[:2])


# In[ ]:

rnn.in_.W.data


# In[ ]:

reporter.observation['model/loss'].data


# In[ ]:

def lossfun(arg1, arg2):
    # calculate loss
    loss = F.sum(model(arg1 - arg2))
    return loss


# In[ ]:

F.MeanSquaredError(y, y)


# In[ ]:

def compute_loss(x_list):
    loss = 0
    for cur_word, next_word in zip(x_list, x_list[1:]):
        loss += model(cur_word, next_word)
    return loss


# #### MNIST

# In[ ]:

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)
        
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


# In[ ]:

train, test = datasets.get_mnist()


# In[ ]:

class MLP(Chain):
    def __init__(self, n_units, n_out):
        super().__init__(  # MLP, self
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),    # n_units -> n_out
        )
        
    def call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

    def __call__(self, x):
        return z.compose(self.l3, F.relu, self.l2, F.relu, self.l1)(x)
        return z.pipe(x, self.l1, F.relu, self.l2, F.relu, self.l3)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y


model = L.Classifier(MLP(100, 10))
optimizer = optimizers.SGD()
optimizer.setup(model)


# In[ ]:

train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)

test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')


# In[ ]:

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
# trainer.extend(extensions.ProgressBar())
trainer.run()  


# In[ ]:

trainer.run()


# In[ ]:

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)


# In[ ]:

l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)
def my_forward(x):
    h = l1(x)
    return l2(h)


# In[ ]:




# In[ ]:

nr.seed(0)
x = Variable(nr.random((2, 4)).astype(np.float32))
x


# In[ ]:

y = my_forward(x)


# In[ ]:

y.data


# In[ ]:

y.backward()


# In[ ]:

x.grad


# In[ ]:

np.finfo(np.float32)


# ### Keras

# In[ ]:

lstm = Sequential([
    LSTM(
        input_dim=P,
        output_dim=4,
        return_sequences=True
    ),
    Dense(output_dim=2),
#     Activation('linear'),
])


lstm.compile(loss="mse", optimizer="rmsprop")


# In[ ]:

epochs = 1
lstm.fit(x13.values[None, :, :], y13, batch_size=32, nb_epoch=epochs, validation_split=0.05)


# In[ ]:

l1, ld, la = lstm.layers
l1.input_dim


# In[ ]:

la


# In[ ]:

la.input_shape


# In[ ]:

x.shape


# In[ ]:

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

_x, _y = create_dataset(xd[['Temp_avg']].values)


# In[ ]:




# In[ ]:

xd[-3:]


# In[ ]:

DataFrame({'X': _x.ravel(), 'Y': _y[:]}).assign(T=xd.Temp_avg.tolist()[:-2])
# _.shape


# In[ ]:

x13[:2]


# In[ ]:

plt.figure(figsize=(16, 10))
xd.query('Yr == 2014').Logcnt.plot()
res0.Logcnt.plot()


# In[ ]:

pf.Poisson


# In[ ]:




# In[ ]:

pf.GAS


# In[ ]:

model2 = pf.GASX(formula='Logcnt~1+ Temp_avg + Dew_avg + Press_avg', data=xd, ar=1, sc=1, family=pf.Skewt())
x = model2.fit()
x.summary()


# In[ ]:

model.plot_fit(figsize=((15,10)))


# ### GPNARX

# In[ ]:

model = pf.GPNARX(xd[['Logcnt']], ar=4, kernel=pf.SquaredExponential())
x = model.fit()
x.summary()


# In[ ]:

model.plot_fit(figsize=((15,10)))


# In[ ]:



