"""
numbers (numpy-related)
"""
from __future__ import annotations

from typing import cast

import numpy as np


def softmax(array: np.ndarray) -> np.float64:
    """
    ソフトマックス関数
    Args:
        array(np.array): 入力
    Returns:
        np.float64: 出力
    """
    array_max: np.float64 = (cast(np.float64, np.max(array)) if np.issubdtype(array.dtype, np.floating)
                             else cast(np.float64, np.max(array.astype(np.float64))))
    exp_x: np.ndarray = np.exp(array - array_max)
    sum_exp_x: np.float64 = cast(np.float64, np.sum(exp_x))
    return exp_x / sum_exp_x


def sigmoid(x: np.float64 | np.ndarray) -> np.float64 | np.ndarray:
    """
    シグモイド関数
    Args:
        x(np.float64 | np.ndarray): 入力
    Returns:
        np.float64 | np.ndarray: 出力
    """
    return np.float64(1.0) / (np.float64(1.0) + np.exp(-x))
