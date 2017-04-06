
from tsfresh import extract_features
from tsfresh import extract_relevant_features
# from tsfresh.utilities.dataframe_functions import impute

los = ends_with('_lo')(xdf)
his = ends_with('_hi')(xdf)

xdfstd = (xdf.set_index('Dt')
          .drop(['Event', 'Cnt', 'Day', 'M', 'Yr', 'Doy'] + los + his, axis=1).iloc[:, :].copy()
          .dropna(axis=0, subset=['Logcnt'])
          .copy()
          )
# print(xdfstd.isnull().sum())
xdfstd = impute(xdfstd).assign(
    Id=1,
    Index=lambda x: np.arange(len(x)).astype(int)
).reset_index(drop=1)
# xdfstd = xdfstd.reset_index(drop=0)
# print(xdfstd.isnull().sum())
y = xdfstd.Logcnt
x = xdfstd.drop('Logcnt', axis=1)

# fts = extract_relevant_features(xdfstd, 'Logcnt', column_id='Id', column_sort='Index')
fts = extract_features(x, column_id='Id', column_sort='Index')
features_filtered = select_features(fts, y)


timeseries_, y_ = load_robot_execution_failures()
efs_ = extract_features(timeseries_, column_id="id", column_sort="time")


##########################################################################
# Batches
##########################################################################
def get_batch(x, y, i, evaluation=False, bptt=np.inf):
    seq_len = min(bptt, len(x) - i)
    data = Variable(x[i:i + seq_len], volatile=evaluation)
    target = Variable(y[i:i + seq_len].view(-1))
    return data, target


def batch_getter(x, y):
    @wraps(get_batch)
    def f(i, evaluation=False, bptt=np.inf):
        return get_batch(x, y, i, evaluation, bptt)
    return f


##########################################################################
# Batches 2
##########################################################################
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


##########################################################################
# Training
##########################################################################
def train(model=None, hidden=None, brange=None, batch_getter=None, optimizer=None, eval=False):
    if hidden is None:
        hidden = model.init_hidden(batch_getter.batch_size)
    tot_loss = 0
    res = []
#     maxnorm = 0
    for br in brange:
        x, y = batch_getter(br)
        optimizer.zero_grad()
        # output, hidden = model(x, hidden)
        output = model(x, hidden)
#         hidden = repackage_hidden(hidden)

        res.append(output.data.squeeze())
        if eval:
            continue

        loss = criterion(output, y.view(-1, 1))
        loss.backward()

        T.nn.utils.clip_grad_norm(model.parameters(), 3)

        norms = [T.norm(p.grad.data) for p in m.parameters()]
        maxnorm = max(norms)
        if maxnorm > train.mnorm:
            train.mnorm = maxnorm
            print('max(grad) = {:.3f}'.format(maxnorm))

        optimizer.step()
        tot_loss += loss
    res = T.stack(res).view(-1).numpy()
    if eval:
        return res
    return tot_loss, res

train.mnorm = 0
# tot_loss, hidden, res = train(model=model, hidden=None, brange=brange, batch_getter=batch_getter, optimizer=optimizer)
# print(tofloat(tot_loss))


# With batch_iter
def train_epoch(xt, yt, model=None, bptt=20, hidden=None, optimizer=None, eval=False):
    if hidden is None:
        hidden = model.init_hidden(batch_getter.batch_size)
    tot_loss = 0
    res = []

    for x, y in batch_iter(xt, y=yt, bptt=bptt, evaluation=eval):
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

        norms = [T.norm(p.grad.data) for p in m.parameters()]
        maxnorm = max(norms)
        if maxnorm > train.mnorm:
            train.mnorm = maxnorm
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


################################################################################
# Train/test
# Cloud_cover
################################################################################

def split(X, y, ratio=.9):
    null = X.isnull().any(axis=1) | y.isnull()
    print(X.shape)
    X = X[~null]
    print(X.shape)
    y = y[~null]

    N = int(len(X) * ratio)
    print(len(X))
    print(N)

    Xr = X[:N]
    yr = y[:N]

    Xs = X[N:]
    ys = y[N:]
    return Xr, yr, Xs, ys

# X = ddf[fts + ['N']]  #.dropna(axis=0)
X = ddf[fts + [null_col]]  #.dropna(axis=0)
Xr, yr, Xs, ys = split(X.drop(null_col, axis=1), X.Cloud_cover, ratio=.5)

rf = RandomForestRegressor(n_estimators=30, oob_score=True).fit(Xr, yr, )
ypred = rf.predict(Xs)

def show_preds(ys, pred):
    plt.plot(ys, pred, '.')
    plt.xlabel('Actual')
    plt.ylabel('Pred')
    plt.savefig('cloud_cover_model_perf.png', bbox_inches='tight')

show_preds(ys, pred)
