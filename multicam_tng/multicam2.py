import numpy as np
from multicam.qt import qt, qt_gauss, qt_gauss_base, qt_inverse_gauss_base
from numpy import ndarray
from scipy.stats import rankdata
from sklearn import linear_model


def multicam_prediction(x: ndarray, x_train: ndarray, y_train: ndarray):
    """MultiCAM algorithm without the class scaffolding for utility. Still very fast as no QT."""
    assert x.ndim == x_train.ndim == y_train.ndim == 2
    assert x_train.shape[0] == y_train.shape[0]

    # ranks are unnecessary to start as qt_gauss already uses 'ordinal'
    xgt = qt_gauss(x_train, axis=0, method="ordinal")
    ygt = qt_gauss(y_train, axis=0, method="ordinal")

    reg = linear_model.LinearRegression()
    reg.fit(xgt, ygt)

    xg = qt_gauss(x, axis=0, method="ordinal")

    yng = reg.predict(xg)
    yngt = reg.predict(xgt)

    yg = qt_gauss_base(yng, yngt)
    yp = qt_inverse_gauss_base(yg, y_train)

    return yp


# def multicam_new_prediction(
#     x: ndarray, x_train: ndarray, y_train: ndarray, y_true: ndarray
# ):
#     """MultiCAM algorithm generalized so that training and output distributions can be different."""
#     assert x.ndim == x_train.ndim == y_train.ndim == 2
#     assert x_train.shape[0] == y_train.shape[0]

#     # no need to convert to ranks as this is done implicitly in `qt_gauss`
#     xgt = qt_gauss(x_train, axis=0, method="ordinal")
#     ygt = qt_gauss(y_train, axis=0, method="ordinal")

#     reg = linear_model.LinearRegression()
#     reg.fit(xgt, ygt)

#     xr = rankdata(x, axis=0, method="ordinal").astype(float)
#     xr *= len(x_train) / len(x)
#     xrt = rankdata(x_train, axis=0, method="ordinal").astype(float)
#     xg = qt_gauss_base(xr, xrt)

#     yng = reg.predict(xg)
#     yngt = reg.predict(xgt)

#     yg = qt_gauss_base(yng, yngt)
#     yp = qt_inverse_gauss_base(yg, y_true)

#     return yp


def multicam_new_prediction(
    x: ndarray, x_train: ndarray, y_train: ndarray, y_true: ndarray
):
    """MultiCAM algorithm generalized so that training and output distributions can be different."""
    assert x.ndim == x_train.ndim == y_train.ndim == 2
    assert x_train.shape[0] == y_train.shape[0]

    # no need to convert to ranks as this is done implicitly in `qt_gauss`
    xgt = qt_gauss(x_train, axis=0, method="ordinal")
    ygt = qt_gauss(y_train, axis=0, method="ordinal")

    reg = linear_model.LinearRegression()
    reg.fit(xgt, ygt)

    xr = rankdata(x, axis=0, method="ordinal").astype(float)
    xr *= len(x_train) / len(x)
    xrt = rankdata(x_train, axis=0, method="ordinal").astype(float)
    xg = qt_gauss_base(xr, xrt)

    yng = reg.predict(xg)

    yp = np.full_like(yng, fill_value=np.nan)
    for ii in range(yng.shape[1]):
        yp[:, ii] = qt(yng[:, ii], y_true[:, ii])

    return yp


def _create_rank_lookup(x):
    assert x.ndim == 2
    n_features = x.shape[1]
    rank_lookup = {}

    # lookup table of ranks
    for jj in range(n_features):
        xjj = np.sort(x[:, jj])
        u, c = np.unique(xjj, return_counts=True)
        lranks = np.cumsum(c) - c + 1
        hranks = np.cumsum(c)
        rank_lookup[jj] = (u, lranks, hranks)

    return rank_lookup


def qt_ranks_base(x: ndarray, x_base: ndarray):
    """Gaussinize input based on another dataset."""
    # always assume second dimension is n_features.
    assert x.ndim == 2 and x_base.ndim == 2
    assert x.shape[1] == x_base.shape[1]
    n_features = x.shape[1]
    x_ranks = np.zeros_like(x) * np.nan
    for jj in range(n_features):
        x_jj = x[:, jj]
        xb_jj = np.sort(x_base[:, jj])  # required for np.interp
        xb_ranks_jj = rankdata(xb_jj, method="ordinal")
        x_ranks[:, jj] = np.interp(x_jj, xb_jj, xb_ranks_jj)
    return x_ranks


def _get_ranks_based(
    x: ndarray, x_base: ndarray, rank_lookup: dict, mode: str = "middle"
):
    assert mode in {"middle", "random"}
    assert x.ndim == 2
    assert x_base.ndim == 2
    n_features = x.shape[1]

    # start by interpolating ranks naively
    xr = qt_ranks_base(x, x_base)

    # if value is in training data, get middle or random rank
    for jj in range(n_features):
        x_jj = x[:, jj]
        uniq, lranks, hranks = rank_lookup[jj]

        in_train = np.isin(x_jj, uniq)
        u_indices = np.searchsorted(uniq, x_jj[in_train])
        lr, hr = lranks[u_indices], hranks[u_indices]  # repeat appropriately
        xr[in_train, jj] = (
            np.random.randint(lr, hr + 1) if mode == "random" else (lr + hr) / 2
        )

    assert np.sum(np.isnan(xr)) == 0

    return xr


def multicam_new_prediction_repeats(
    x: ndarray, x_train: ndarray, y_train: ndarray, y_true: ndarray
):
    """MultiCAM algorithm without the class scaffolding for utility. Still very fast as no QT."""
    assert x.ndim == x_train.ndim == y_train.ndim == 2 == y_true.ndim
    assert x_train.shape[0] == y_train.shape[0]
    assert x.shape[1] == x_train.shape[1]
    assert y_train.shape[1] == y_true.shape[1]
    n_targets = y_train.shape[1]

    # qt_gauss already uses 'ordinal' and breaks repeats in training data
    xgt = qt_gauss(x_train, axis=0)
    ygt = qt_gauss(y_train, axis=0)

    reg = linear_model.LinearRegression()
    reg.fit(xgt, ygt)

    # TODO: handle values that are outside the domain of training
    # want to assign gaussianized values to `x` that are as close as possible to `xgt`

    # the next two lines are only necessary to account for repeats in features
    rank_lookup = _create_rank_lookup(x_train)
    xr = _get_ranks_based(x, x_train, rank_lookup, mode="middle")
    xrt = rankdata(x_train, axis=0, method="ordinal")
    xg = qt_gauss_base(xr, xrt)  # should be exactly the same if value is contained!

    yg = reg.predict(xg)

    # we want to avoid repeats 'bunching up' to reproduce correct output distribution in ALL cases.
    # does not change rank order otherwise
    # TODO: qt internally may already take care of this... but it's good to know that it is needed
    ygf = qt_gauss(yg, axis=0, method="ordinal")

    # KEY: finally we want to reproduce some final 'true' distribution
    # so we abundance match each corresponding target variable outputed from the LR prediction
    yp = np.full_like(ygf, fill_value=np.nan)
    for ii in range(n_targets):
        yp[:, ii] = qt(ygf[:, ii], y_true[:, ii])  # correctly interpolates

    return yp
