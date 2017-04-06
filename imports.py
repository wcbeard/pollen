from collections import OrderedDict, defaultdict
from functools import partial, wraps
from importlib import reload
import itertools as it
import os
# import shutil
import string
import uuid
import re
import datetime as dt
import time

import numpy.random as nr
import numpy as np
import feather
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.compat import lfilter, lmap, lrange, lzip
from pandas import pandas as pd, DataFrame, Series

import toolz.curried as z

pd.options.display.notebook_repr_html = False
pd.options.display.width = 120
pd.options.mode.use_inf_as_null = True
pd.options.display.max_columns = 50
