import numpy as np
import ctypes
import os
import fnmatch
from collections import namedtuple

dir_path = os.environ["LIBMF_OBJ"] if "LIBMF_OBJ" in os.environ \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for f in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, f)) and fnmatch.fnmatch(f, '*libmf*.so'):
        compiled_src = os.path.join(dir_path, f)
        break

mf = ctypes.CDLL(compiled_src)
c_float_p = ctypes.POINTER(ctypes.c_float)

''' libmf enums '''

P_L2_MFR = 0
P_L1_MFR = 1
P_KL_MFR = 2
P_LR_MFC = 5
P_L2_MFC = 6
P_L1_MFC = 7
P_ROW_BPR_MFOC = 10
P_COL_BPR_MFOC = 11

RMSE = 0
MAE = 1
GKL = 2
LOGLOSS = 5
ACC = 6
ROW_MPR = 10
COL_MPR = 11
ROW_AUC = 12
COL_AUC = 13

Option = namedtuple('Option', ['name', 'type', 'value'])

default_options = [
    Option(name="fun", type=ctypes.c_int, value=P_L2_MFR),
    Option(name="k", type=ctypes.c_int, value=8),
    Option(name="nr_threads", type=ctypes.c_int, value=12),
    Option(name="nr_bins", type=ctypes.c_int, value=20),
    Option(name="nr_iters", type=ctypes.c_int, value=20),
    Option(name="lambda_p1", type=ctypes.c_float, value=0.0),
    Option(name="lambda_p2", type=ctypes.c_float, value=0.1),
    Option(name="lambda_q1", type=ctypes.c_float, value=0.0),
    Option(name="lambda_q2", type=ctypes.c_float, value=0.1),
    Option(name="eta", type=ctypes.c_float, value=0.1),
    Option(name="do_nmf", type=ctypes.c_bool, value=False),
    Option(name="quiet", type=ctypes.c_bool, value=False),
    Option(name="copy_data", type=ctypes.c_bool, value=False),
]

''' libmf enums '''


class MFModel(ctypes.Structure):
    _fields_ = [("fun", ctypes.c_int),
                ("m", ctypes.c_int),
                ("n", ctypes.c_int),
                ("k", ctypes.c_int),
                ("b", ctypes.c_float),
                ("P", c_float_p),
                ("Q", c_float_p)]


class MFParam(ctypes.Structure):
    _fields_ = [(opt.name, opt.type) for opt in default_options]


options_ptr = ctypes.POINTER(MFParam)


class MF(object):
    def __init__(self, *args, **kwargs):
        self.model = None
        self._options = MFParam()
        for kw in kwargs:
            if kw not in [opt.name for opt in default_options]:
                print("Unrecognized keyword argument '{0}={1}'".format(kw, kwargs[kw]))

        for opt in default_options:
            value = kwargs[opt.name] if opt.name in kwargs else opt.value
            setattr(self._options, opt.name, opt.type(value))

    def mf_predict(self, X):
        """
        assuming we have already run the fit method, predict the values at certain indices of the data matrix
        :param X: (n, 2) shaped numpy array
        :return: numpy array of length n
        """
        if self.model is None:
            return LookupError("no model data is saved, try running model.mf_fit(...) first")
        ensure_width(X, 2)
        nnx = X.shape[0]
        out = np.zeros(nnx)
        out = out.astype(np.float32)
        X = X.astype(np.float32)
        X_p = X.ctypes.data_as(c_float_p)
        nnx_p = ctypes.c_int(nnx)
        mf.pred_model_interface(nnx_p, X_p, ctypes.c_void_p(out.ctypes.data), ctypes.byref(self.model))
        return out

    def mf_fit(self, X):
        """
        factorize the i x j data matrix X into (j, k) (k, i) sized matrices stored in MF.model
        :param X: (n, 3) shaped numpy array [known index and values of the data matrix]
        """
        ensure_width(X, 3)
        d = X.astype(np.float32)
        data_p = d.ctypes.data_as(c_float_p)
        nnx = ctypes.c_int(X.shape[0])
        mf.fit_interface.restype = ctypes.POINTER(MFModel)
        mf.fit_interface.argtypes = (ctypes.c_int, c_float_p, options_ptr)
        out = mf.fit_interface(nnx, data_p, self._options)
        self.model = out.contents

    def mf_cross_validation(self, X, folds=5):
        """
        :param X: (n, 3)
        :param folds: number of train / test splits
        :return: average score across all folds
        """
        ensure_width(X, 3)
        d = X.astype(np.float32)
        data_p = d.ctypes.data_as(c_float_p)
        nnx = ctypes.c_int(X.shape[0])
        mf.cross_valid_interface.restype = ctypes.c_double
        mf.cross_valid_interface.argtypes = (ctypes.c_int, c_float_p, options_ptr, ctypes.c_int)
        score = mf.cross_valid_interface(nnx, data_p, self._options, folds)
        return score

    def mf_train_test(self, X, V):
        ensure_width(X, 3)
        ensure_width(V, 3)
        nnx = ctypes.c_int(X.shape[0])
        nnx_valid = ctypes.c_int(V.shape[0])

        train_p = X.astype(np.float32)
        train_p = train_p.ctypes.data_as(c_float_p)

        test_p = V.astype(np.float32)
        test_p = test_p.ctypes.data_as(c_float_p)

        mf.train_valid_interface.restype = ctypes.POINTER(MFModel)
        mf.train_valid_interface.argtypes = (ctypes.c_int, ctypes.c_int, c_float_p, c_float_p, options_ptr)
        out = mf.train_valid_interface(nnx, nnx_valid, train_p, test_p, self._options)
        self.model = out.contents


def ensure_width(x, width):
    if x.shape[1] != width:
        raise ValueError("must be sparse array of shape (n, {0})", width)


def generate_test_data(xs, ys, k, indices_only=False):
    rx = np.random.random_integers(0, xs, k)
    ry = np.random.random_integers(0, ys, k)
    rv = np.random.rand(k)
    return np.vstack((rx, ry, rv)).transpose() if not indices_only else np.vstack((rx, ry)).transpose()
