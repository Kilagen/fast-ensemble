from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd


def to_pandas(X: Any) -> pd.DataFrame:
    try:
        return pd.DataFrame(X)
    except ValueError:
        raise ValueError(
            "Wrong dtype. Expected ndarray (structured or homogeneous), Iterable, dict, or DataFrame, found {}".format(type(X))
        )


def average_preds(preds_array):
    return sum(preds_array) / len(preds_array)
