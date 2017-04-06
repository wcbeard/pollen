import datetime as dt
from functools import wraps
import requests
import time

import pandas as pd
import numpy as np
import numpy.random as nr
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch as T
import toolz.curried as z


toyear, tomonth = dt.date.today().year, dt.date.today().month
yrmths = [(yr, m) for yr in range(2000, 2018)
          for m in range(1, 13)
          if (yr, m) <= (toyear, tomonth)]
flatten_multindex = lambda xs: [
    '_'.join([lvl1, lvl2.lower()]) if lvl2 else lvl1 for lvl1, lvl2 in xs]
# mse = lambda x, y: np.sqrt(((x - y) ** 2).sum())
mse = lambda x, y: (((x - y) ** 2).sum()) / len(x)


def check_one2one(df, k1, krest):
    krest = [krest] if isinstance(krest, str) else krest
    assert df.groupby(
        [k1] + krest).size().reset_index(drop=0).groupby(k1).size().eq(1).all()


def random_requests(urls, mean=2, sd=2, min=.1):
    "Random intervals between sequential requests."
    for u in urls:
        r = requests.get(u)
        if r.from_cache:
            continue
        sleep_time = max((np.random.randn() + mean) * sd, min)
        time.sleep(sleep_time)


def impute(df):
    df = df.copy()
    for c in df:
        s = df[c]
        bm = ~(s == s)

        df.loc[bm, c] = s.median()
    return df


def ends_with(suff):
    return (lambda xs: [x for x in xs if x.endswith(suff)])


######################################################################
# Batches 1
######################################################################
def batchify(x, batch_size=5, train_len=100):
    """Cut a sequence x into smaller `train_len`-long sequences.
    Then return a list where each element contains
    `batch_size` of these sequences.
    """
    # Create extra dimension; as if making a list
    # of the `train_len`-long mini-sequences
    seq_ixs = np.arange(0, len(x), train_len)[:-1]
    batchified = np.stack([x[six:six + train_len] for six in seq_ixs])

    batch_ixs = np.arange(0, len(batchified), batch_size)[:-1]
    return [batchified[bix:batch_size] for bix in batch_ixs]


def test_batchify():
    x = np.array(
        [[0, 1],
         [2, 3],
         [4, 5],
         [6, 7],
         [8, 9],
         [10, 11],
         [12, 13],
         [14, 15],
         [16, 17],
         [18, 19]])
    [only_batch] = batchify(x, batch_size=2, train_len=3)
    subseq1, subseq2 = only_batch

    shouldbe1 = np.array(
        [[0, 1],
         [2, 3],
         [4, 5]])
    assert (subseq1 == shouldbe1).all()

    shouldbe2 = np.array(
        [[6, 7],
         [8, 9],
         [10, 11]])
    assert (subseq2 == shouldbe2).all()


######################################################################
# Batches 2
######################################################################
def to_sub_seqs(x: np.array, seq_len=5, warn=True):
    ixs = np.arange(0, len(x) + 1, seq_len)[:-1]
    subseqs = [x[i:i + seq_len] for i in ixs]
    res = np.array(subseqs)
    to_sub_seqs.diff = diff = x.size - res.size
    if warn and diff:
        print('{} elems dropped from end'.format(diff))
    return res


def test_to_sub_seqs():
    for i in range(8, 12):
        res = to_sub_seqs(np.arange(i), 5, warn=False)
        assert to_sub_seqs.diff == (i % 5)
        assert len(res) == i // 5


def batch_getterer(x, *, y=None, batch_size=5, var=True):
    batch_ixs = np.arange(0, len(x) + 1, batch_size)[:-1]
    nslices = len(batch_ixs) * batch_size
    print('nslices', nslices, 'len(x)', len(x))
    if nslices < len(x):
        print('Warning: {} sequences clipped'.format(len(x) - nslices))

    def get_batch(x, i):
        ix = batch_ixs[i]
        res = x[ix:ix + batch_size]
        if var:
            return Variable(res)
        return res

    def batch_getter(i):
        xres = get_batch(x, i)
        if y is None:
            return xres
        return xres, get_batch(y, i)

    batch_getter.batch_size = batch_size
    return batch_getter, range(len(batch_ixs))


##########################################################################
# Batches 3
##########################################################################
isint = lambda x: np.issubdtype(type(x), int)


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
            batches = self.x[bixs:bixs + self.seq_len]
        else:
            batches = [self.x[bix:bix + self.seq_len] for bix in bixs]
        return self.retfunc(batches)

    idxmax = property(lambda x: len(x.ix) - 1)


class BatchArray(BatchArraySingle):

    def __init__(self, x=None, y=None, seq_len=5, truncate=True, tovar=True, batch_size=20):
        if seq_len is None:
            seq_len = len(y)
        super().__init__(x=x, seq_len=seq_len, truncate=truncate, tovar=tovar)

        self.xb = BatchArraySingle(
            x=x, seq_len=seq_len, truncate=truncate, tovar=tovar)
        if truncate:
            self.rem = BatchArray(x=x, y=y, seq_len=seq_len,
                                  truncate=False, tovar=tovar)

        self.y = y
        self.yb = BatchArraySingle(
            x=y, seq_len=seq_len, truncate=truncate, tovar=tovar)
        self.batch_size = batch_size
        self.num_batches = len(self.ix) // batch_size
        self.num_truncated_rows = self.num_batches * batch_size * seq_len
        self.num_leftover_rows = len(self.y) - self.num_truncated_rows

    def __getitem__(self, ix):
        return self.xb.__getitem__(ix), self.yb.__getitem__(ix)

    def batch_ix_iter(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        nb = len(self.ix) // batch_size
        return np.arange(nb * batch_size).reshape(-1, batch_size)

    @property
    def train_samples_y(self):
        return self.y[:self.num_truncated_rows].numpy().ravel()
        return np.concatenate([self[bix][1].data.numpy().ravel() for bix in
                               self.batch_ix_iter()])

    @property
    def train_samples_x(self):
        return self.x[:self.num_truncated_rows]

    @property
    def test_samples_x(self):
        return self.x[self.num_truncated_rows:]

    @property
    def test_samples_y(self):
        return self.y[self.num_truncated_rows:]


######################################################################
# Tensor manip
######################################################################
def collapse_dims_(t, dim_prods: [(int,)]):
    '''tens[a x b x c], [(a, b), (c,)]
    -> tens[a]
    '''
    dims = t.size()
    new_dims = [int(np.prod([dims[i] for i in dimtup]))
                for dimtup in dim_prods]
    # new_dims = lmap(np.prod, new_dims)
    return new_dims


def collapse_dims(t, dim_prods: [(int,)]):
    new_dims = collapse_dims_(t, dim_prods=dim_prods)
    # print(new_dims)
    return t.contiguous().view(*new_dims)


def test_collapse_dims_():
    assert collapse_dims_(T.randn(3, 4, 5), [(0, 1), (2,)]) == [12, 5]
    assert collapse_dims_(T.randn(3, 4, 5), [(0,), (1, 2,)]) == [3, 20]


def ravel(t):
    try:
        return t.view(-1, t.size(-1))
    except RuntimeError as e:
        return ravel(t.contiguous())


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


##########################################################################
# Validation Utilities
##########################################################################
def add_dates_(df, l):
    """Batches will have `l` leftover elements.
    I stripped out the dates from the features,
    and just left day of the year integers, so this function
    reconstructs the actual dates from these day of year features.
    """
    ldays = df.iloc[-2 * l:].assign(
        Yr=lambda x: (x.Doy.shift() > x.Doy  # New year when
                      ).fillna(False).astype(int).cumsum()  # .add(2000)
    )
    ldays['Yr'] = ldays['Yr'] + (2017 - ldays['Yr'].max())
    ldays['Dt'] = np.concatenate([pd.date_range('{}-01-01'.format(yr), '{}-12-31'.format(yr), freq='D')[gdf.Doy - 1].date
                                  for yr, gdf in ldays.groupby(['Yr'], sort=False)])

    ld1, ld2 = np.array_split(ldays, 2)
    return ld1, ld2, ldays


##########################################################################
# Processing
##########################################################################
def rep_with_dummies_(df, col):
    df = df.copy()
    newcs = pd.get_dummies(df[col]).astype(int)
    for c in newcs:
        df[c] = newcs[c]
    return df.drop(col, axis=1)


def replace_with_dummies(df, cols):
    """Return a copy of df w/ each of `cols` replaced
    with its dummified self"""
    for c in cols:
        df = rep_with_dummies_(df, c)
    return df


def filter_dtypes(df, dtypes=[float]):
    cs = [k for k, v in df.dtypes.items() if any(v == d for d in dtypes)]
    return cs


def log_(s):
    lg = np.log10(s)
    lg[np.isneginf(lg)] = 0
    return lg


def join_pollen_weather(poldf, ddf, time_cols, ycol='Logcnt'):
    pol_joinable = poldf.set_index(
        'Date')[['Count', 'Prev_cnt', 'Prev_cnt_null']].ix[ddf.Dt.dt.date]

    xdf = (ddf.assign(**{
        c: pol_joinable[c].tolist() for c in pol_joinable
    }).assign(
        Logcnt=lambda x: log_(x.Count),
        Log_prev_cnt=lambda x: log_(x.Prev_cnt),
    ).dropna(axis=0)
        .assign(
        Day_diff=lambda x: (x.Day_int - x.Day_int.shift(1)
                            ).fillna(1).sub(1).astype(int)
    ))
    xdf.Prev_cnt_null = xdf.Prev_cnt_null.astype(int)

    feats = filter_dtypes(
        ddf, dtypes=[int, float]) + ['Log_prev_cnt', 'Prev_cnt_null', 'Day_diff']
    feats = np.setdiff1d(feats, time_cols + ['Y']).tolist()

    rxdf = xdf[feats].copy()
    ry = xdf[ycol].astype(np.float32)

    ss = StandardScaler()
    rx = ss.fit_transform(rxdf).astype(dtype=np.float32)

    xt = T.from_numpy(rx)
    yt = T.from_numpy(ry.values[:, None])
    return xdf, xt, yt, rx, rxdf, ry


######################################################################
# fake data
######################################################################
def gen_dat1(P=3, N=20, dtype=np.float32):
    x = nr.randint(0, high=5, size=(N, P)).astype(dtype)
    sm = x.sum(axis=1)
    y = sm + np.r_[0, sm[:-1]]
    return x, y[:, None]


def gen_dat2(P=3, N=20, dtype=np.float32):
    x = nr.randint(-5, high=5, size=(N, P)).astype(dtype)
    return x, x.sum(axis=1).cumsum()[:, None]


######################################################################
# random stuff
######################################################################
def check_cached(min_time):
    def deco(f):
        @wraps(f)
        def wrapper(*a, **k):
            t = time.perf_counter()
            res = f(*a, **k)
            e = time.perf_counter() - t
            if e > min_time:
                print('{:.2f}s elapsed'.format(e))
            return res
        return wrapper
    return deco


def show_dt(X, y, criterion='gini', max_depth=4):
    "Run decision tree on data, output the graph, and open it"
    from sklearn.tree import DecisionTreeClassifier, export_graphviz

    dtc = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth).fit(X, y)
    export_graphviz(dtc, feature_names=X.columns, out_file='tree.dot')
    get_ipython().system('dot -Tpng tree.dot -o tree.png')
    get_ipython().system('open tree.png')


def read(fn):
    with open(fn, 'r') as f:
        return f.read()
