"""A module containing a labeling models."""

import numpy as np
import numpy.typing as npt

from openseize.producer


class PowerThreshold:
    """A labeling model for prelabeling EEG data into Wake, REM
    Non-REM states or Artifact.

    This model compares the band power in the delta and theta bands and if
    present the total EMG power to label EEG data.
    """

    def __init__(
        self,
        fs: float,
        wintime: float = 4,
        axis: int = -1,
    ) -> Self:
        """Initialize this Model.

        Args:
            fs:
                The sampling rate of the EEGs.
            wintime:
                The size of the sample window for power estimation.
            axis:
                The sample axis of the EEGs/EMGs.

        Returns:
            None
        """

        self.fs
        self.winsize = wintime * fs
        self.axis = axis

    def estimate(
        self,
        eeg: npt.NDArray | Producer,
        emg: Optional[npt.NDArray | Producer],
        method: callable = otsu
    ) -> list[float], npt.NDArray, npt.NDArray:
        """Estimates the optimal delta, theta and optionally EMG powers
        that partition an EEG into Wake, REM, NREM or Artifact.




