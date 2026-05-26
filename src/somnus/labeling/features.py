"""A collection of functions that return features for training and predicting
with somnus' models."""

import numpy as np
import numpy.typing as npt
import scipy.signal as sps



def path_length(
    arr: npt.NDArray[np.float64],
    axis: int = -1,
) -> float | npt.NDArray:
    """Returns the path length from a 2D array of coordinate positions.

    Args:
        arr:
            An 2D array of coordinate positions with samples along axis.
        axis:
            The sample axis of arr.

    Returns:
        A float or an array of floats.

    Raises:
        A ValueError is issued if arr contains np.nan float type.
    """

    if np.any(np.isnan(arr)):
        msg = 'arr argument must not contain np.nan float type.'
        raise ValueError(msg)

    ds = np.sqrt(np.sum(np.diff(arr, axis=-1) ** 2, axis=-2))
    sum_ds: npt.NDArray[np.float64] = np.sum(ds, axis=-1)

    return np.squeeze(sum_ds)


def bandpowers(
    arr: npt.NDArray[np.float64],
    fs: float,
    bands: list[tuple[float, float]],
    axis: int = -1,
    reducer: func # to combine across EEG channels someway like median etc
    estimator: sps.welch,
    **kwargs,
) -> float:
    """ """

    pass

