from multicam.qt import qt_gauss, qt_gauss_base, qt_inverse_gauss_base
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
