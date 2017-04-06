
# coding: utf-8

# In[ ]:

from imports import *

# %mkdir cache
import joblib; mem = joblib.Memory(cachedir='cache')

import toolz.curried as z

import torch as T
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from functools import wraps

pd.options.display.max_columns = 50
get_ipython().magic('matplotlib inline')


# In[ ]:

import util.pollen_utils; reload(util.pollen_utils); from util.pollen_utils import (
    pollen_url, pollen_date2df as pollen_date2df_, pollen_data as pollen_data_,
    parse_pollen_page
)

import util.utils; reload(util.utils); from util.utils import (
    check_one2one, yrmths, flatten_multindex, ends_with,
    collapse_dims, test_collapse_dims_,
    BatchArray,
    ravel, repackage_hidden, mse
)
date = lambda xs: dt.datetime(*xs)


pollen_date2df = mem.cache(pollen_date2df_)
pollen_data = mem.cache(pollen_data_)

test_collapse_dims_(T)
(";")


# ## Load

# In[ ]:

rx = feather.read_dataframe('cache/x.fth').values
rx_ = feather.read_dataframe('cache/x_.fth')
ry = feather.read_dataframe('cache/y.fth').iloc[:, 0]


# ## Model

# In[ ]:

class Rnn(nn.Module):
    def __init__(self, P=3, nhidden=21, num_layers=1, dropout=0):
        super().__init__()
        self.P, self.nhidden, self.num_layers, self.dropout = (
            P, nhidden, num_layers, dropout
        )
        self.rnn = nn.GRU(P, nhidden, num_layers, batch_first=True, dropout=dropout)
        self.Dropout = nn.Dropout(p=dropout)
        self.decoder = nn.Linear(nhidden, 1)
        self.init_weights()
        self.zero_grad()

    def __dir__(self):
        return super().__dir__() + list(self._modules)
    
    def forward(self, input, hidden=None, outputh=False):
        if hidden is None:
            hidden = self.hidden
        out1, hout = self.rnn(input, hidden)
        out1d = self.Dropout(out1)
        out2 = self.decoder(ravel(out1d))
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


# ## Train

# In[ ]:

def train_epoch(barray, model=None, hidden=None, optimizer=None, eval=False, batch_size=None):
    batch_size = batch_size or barray.batch_size
    assert batch_size or hidden
    if hidden is None:
        hidden = model.init_hidden(batch_size)
    res = []
    global x, y, output, loss
    ss, _n = 0, 0

    for bix in barray.batch_ix_iter(batch_size=batch_size):
        x, y = barray[bix]
        optimizer.zero_grad()
        output = model(x, hidden)
        
        res.append(output.data.squeeze())
        if eval:
            continue

        loss = criterion(output, y.view(-1, 1))
        
        loss.backward()

        T.nn.utils.clip_grad_norm(model.parameters(), 3)

        maxnorm = max(T.norm(p.grad.data) for p in model.parameters())
        if maxnorm > train_epoch.mnorm:
            train_epoch.mnorm = maxnorm
            print('max(grad) = {:.3f}'.format(maxnorm))

        optimizer.step()
        
        ss += tofloat(loss) * len(output)  # keep track of ss
        _n += len(output)

    res = T.stack(res).view(-1).numpy()
    if eval:
        return res
    tot_loss = ss / _n
    return tot_loss, res
     
train_epoch.mnorm = 0
# tot_loss, hidden, res = train(model=model, hidden=None, brange=brange, batch_getter=batch_getter, optimizer=optimizer)
# print(tofloat(tot_loss))


# In[ ]:




# In[ ]:

VALFN = '/tmp/res/val.txt'
TRNFN = '/tmp/res/trn.txt'

def report_hook(model, res, vals=None):
    print()
    val_pred(model, warmup=True)
    yspred, ys = val_pred(model, warmup=False)
    val_acc = mse(yspred, ys)
    vals.append(val_acc)
    trn_acc = mse(trn_y, res)

    with open(VALFN, 'a') as f:
        f.write('{:}\n'.format(val_acc))
    with open(TRNFN, 'a') as f:
        f.write('{:}\n'.format(trn_acc))
    print('{:,.3f}; val: {:,.4f}'.format(trn_acc, val_acc), end='; ')

    
def train_epochs(model, optimizer=None, rng=(500, ), print_every=10, report_hook=None, report_kw={}):
    with open(VALFN, 'w') as f: pass
    with open(TRNFN, 'w') as f: pass
    vals = []
    
    for i in range(*rng):
        _, res = train_epoch(ba, model=model, hidden=None, optimizer=optimizer)

        print('.', end='')
        if i % print_every == 0:
            if report_hook:
                report_hook(model, res, vals=vals)
        
    return res, min(vals)


# ## Valid

# In[ ]:

def add_dates(df, l):
    """Batches will have `l` leftover elements.
    I have stripped out the dates from the features,
    and just left day of the year integers, so this function
    reconstructs the actual dates from these day of year features.
    """
    ldays = df.iloc[-2 * l:].assign(
        Yr=lambda x: (x.Doy.shift() > x.Doy  # New year when
                      ).fillna(False).astype(int).cumsum()  #.add(2000)
    )
    ldays['Yr'] = ldays['Yr'] + (2017 - ldays['Yr'].max())
    ldays['Dt'] = np.concatenate([pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')[gdf.Doy - 1].date
                                  for yr, gdf in ldays.groupby(['Yr'], sort=False)])

    ld1, ld2 = np.array_split(ldays, 2)
    return ld1, ld2, ldays


# In[ ]:

def val_pred(model, warmup=True):
    if warmup:
        model.set_hidden(1)

    ix = int(not warmup)
    Dt = baval.Dt[ix]
    xs, ysv = baval[[ix]]

    ys = Series(unvar(ysv), index=Dt)
    yspred = model(xs)

    yspred_s = Series(unvar(yspred), index=Dt)
    return yspred_s, ys


# ## Setup

# In[ ]:

xt = T.from_numpy(rx)
yt = T.from_numpy(ry.values[:, None])
print(xt.size())
print(yt.size())

criterion = nn.MSELoss()


# In[ ]:

seq_len = 25
bcz = 32
# bcz = 64
# bcz = 128
# bcz = 512
ba = BatchArray(x=xt, y=yt, seq_len=seq_len, batch_size=bcz)
trn_y = np.concatenate([ba[bix][1].data.numpy().ravel() for bix in ba.batch_ix_iter()])

L = len(trn_y)
l = len(xt) - L
print('L=', L, 'l=', l)
ld1, ld2, _ = add_dates(rx_, l)

baval = BatchArray(x=xt[-2 * l:], y=yt[-2 * l:], seq_len=l, batch_size=1)
baval.Dt = [ld1.Dt, ld2.Dt]


# In[ ]:

tofloat = lambda x: x.data[0]
unvar = lambda x: x.data.numpy().ravel()


# In[ ]:

def opt_func(x):
    print('==> Starting', x)
    nhidden, num_layers, dropout, lr, opt_method = x
    n_iters = 50
    model = Rnn(P=rx.shape[-1], nhidden=nhidden, num_layers=num_layers, dropout=dropout)
    
    opter = getattr(optim, opt_method)      # RMSprop, Adam
    optimizer = opter(model.parameters(), lr=lr)
    
    vals = []
    
    st = time.perf_counter()
    res, mvals = train_epochs(model=model, optimizer=optimizer, rng=(n_iters, ),
                         print_every=5, report_hook=report_hook,
                         report_kw={'vals': vals}
                        )
    tt = time.perf_counter() - st

    print('\n\nTime: {:.2f}'.format(tt))
    print('Acc: {:.2f}; Val: {:.3f}'.format(mse(res, trn_y), mvals))
    return mvals


# In[ ]:

res = opt_func(8, 1, dropout=0, lr=0.0025, n_iters=50)


# In[ ]:

from skopt import gp_minimize


# In[ ]:

gp_minimize()


# In[ ]:

res = gp_minimize(opt_func,                  # the function to minimize
                  [
                      [8, 16, 32, 64, 128],     # nhidden
                      [1, 2, 3],   # num_layers
                      [.01, .05, .1, .15, .2, .5],              # dropout
                      [.0005, .001, .002, .0025, .005], # lr
                      ['RMSprop', 'Adam'],                # opt_method
                  ],      
#                   acq_func="EI",      # the acquisition function
                  n_calls=150,         # the number of evaluations of f 
                  n_random_starts=5,  # the number of random initialization points
#                   noise=0.1**2,       # the noise level (optional)
                  random_state=123)   # the random seed


# In[ ]:

res


# In[ ]:

# opt_func([64, 3, .01, 0.0050, 'Adam'])
# opt_func([64, 3, .05, 0.0010, 'Adam'])
opt_func([128, 2, .05, 0.001, 'Adam'])  # => good
# opt_func([32, 3, .05, 0.002, 'Adam'])  # => 


# In[ ]:

df = (DataFrame(res['x_iters'], columns='Nhidden Num_layers Dropout Lr Opt_method'.split())
      .assign(Dropout=lambda x: x.Dropout.mul(100).round(1))
     )
df['Y'] = res['func_vals']


# In[ ]:

dfw = df.sort_values('Y', ascending=True).reset_index(drop=1)
feather.write_dataframe(dfw, 'cache/skopt_res.fth')


# In[ ]:

dfw.to_csv('/tmp/sko.csv')


# In[ ]:

dfw.groupby('Dropout').Y.mean()


# In[ ]:

dfw.groupby('Lr').Y.mean()


# In[ ]:

dfw.groupby('Nhidden').Y.mean()


# In[ ]:

dfw.groupby('Num_layers').Y.mean()


# In[ ]:




# In[ ]:

dfw.groupby(['Nhidden', 'Num_layers', ]).Y.mean().plot()


# In[ ]:

dfw.groupby('Opt_method').Y.mean()


# In[ ]:

df.sort_values('Y', ascending=True)[:40]


# In[ ]:

res['func_vals']
res['x_iters']


# In[ ]:

nhidden = 32
num_layers = 2
model = Rnn(P=rx.shape[-1], nhidden=nhidden, num_layers=num_layers, dropout=.5)
model.set_hidden(bcz)
# m = model = Rnn()

optimizer = optim.Adam(model.parameters(), lr = 0.0025)
# optimizer = optim.RMSprop(model.parameters(), lr = 0.002)
model


# In[ ]:

30
30
30
16


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



















# In[ ]:









# In[ ]:








