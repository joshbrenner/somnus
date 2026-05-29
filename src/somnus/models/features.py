"""A collection of functions that return features for training and predicting
with somnus' models."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import scipy.signal as sps
from openseize.core.producer import Producer
from openseize.spectra import metrics


def path_length(
    arr: npt.NDArray[np.float64],
    axis: int = -1,
) -> float | npt.NDArray:
    """Returns the path length from a 2D array of coordinate positions.

    Args:
        arr:
            A 2D array of coordinate positions with samples along axis.
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
    stft: npt.NDArray[np.complex64],
    freqs: npt.NDArray[np.float64],
    bands: list[tuple[float, float]],
    normalize: bool = True,
    reducer: Callable[[npt.NDArray], npt.NDArray] | None = np.max,
) -> float:
    """Returns the power in each band for each time window of the STFT.

    Args:
        stft:
            A 3D array of complex STFT values. The axes of this array are
            expected to be signals, frequencies and time in order.
        frequencies
            The frequency values at which the STFT was estimated.
        bands:
            A list of start, stop tuples for each band over which power is
            computed.
        normalize:
            A boolean indicating if the band powers should be normalized to the
            total power at each time-point in the STFT.
        reducer:
            A function for combining the band powers from each signal in arr.
            The default is to take the maximum band-power across the signals in
            arr.

    Returns:
        An array of shape len(bands) x channels x time
    """


    psds = np.abs(stft) ** 2
    total = metrics.power(psds, freqs, axis=1) if normalize else 1
    results = np.zeros((len(bands), psds.shape[0], psds.shape[-1]))
    for idx, (start, stop) in enumerate(bands):
        results[idx] = metrics.power(psds, freqs, start, stop, axis=1) / total

    return results


if __name__ == '__main__':

    from pathlib import Path
    from openseize import producer
    from openseize.file_io import edf
    from openseize.resampling.resampling import downsample
    from openseize.spectra.estimators import stft
    import matplotlib.pyplot as plt

    fp = '/media/matt/Magnus/data/somnus/C1512_2025-10-13_09_19_40.edf'
    reader = edf.Reader(fp)
    pro = producer(reader, chunksize=10e6, axis=-1)
    x = downsample(pro, M=50, fs=5000, chunksize=10e6).to_array()
    freqs, time, sxx = stft(x, fs=100, resolution=0.125)

    bpowers = bandpowers(sxx, freqs, bands=[(0.5, 4), (4, 8)])
    # TODO need to think about how to combine ch info
    bpowers = np.median(bpowers, axis=1)
    fig, ax = plt.subplots()
    # plot delta power
    ax.plot(time, bpowers[0, :], label='Norm. delta Power', alpha=0.75)
    ax.plot(time, bpowers[1, :], label='Norm. theta Power', alpha=0.75)
    ax.set_title(Path(fp).name)
    ax.set_xlabel('time (s)')
    ax.legend()
    plt.show()

