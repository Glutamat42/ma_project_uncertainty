import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mse_extreme_angles(data, max_angle=80):
    """ returns MSE of the extreme values (greater than max_angle - 20)

    Args:
        data:
        max_angle:

    Returns:

    """
    df = data.copy()
    df = df[abs(df['target_as_angle']).gt(max_angle - 20)]
    return mean_squared_error(df['prediction'], df['target']) if len(df) > 0 else -1


def mse_equally_weighted(data, max_angle=80):
    """ returns MSE equally weighted over all bins

    Args:
        data:

    Returns:

    """
    df = data.copy()
    bins = [-1] + list(range(2, max_angle - 1, 2)) + [360]
    df['angle_bins'] = pd.cut(df['target_as_angle'].abs(), bins=bins)
    mse_per_bin = df.groupby('angle_bins').apply(lambda x: mean_squared_error(x.target, x.prediction) if len(x) > 0 else None)
    mse_per_bin = mse_per_bin[~np.isnan(mse_per_bin)]

    equally_weighted = np.average(mse_per_bin)
    return equally_weighted


def mse(data):
    return mean_squared_error(data.target, data.prediction)


def mae(data):
    return mean_absolute_error(data.target, data.prediction)


def rmse(data):
    return mean_squared_error(data.target, data.prediction, squared=False)


def relative_overfitting(validation_mse, train_mse):
    return (validation_mse - train_mse) / train_mse
