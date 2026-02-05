"""Type definitions used throughout the clophfit package.

It defines the following types:

- ImArray: for float or int images.

- ImMask: for bool image masks.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# or derive class ImSequence(np.ndarray):  # or extend NDArray manually if needed
# ...


ImMask: TypeAlias = NDArray[np.bool_]

ImVector: TypeAlias = (
    NDArray[np.float32] | NDArray[np.int32] | NDArray[np.uint16] | NDArray[np.uint8]
)
ImFrame: TypeAlias = (
    NDArray[np.float32] | NDArray[np.int32] | NDArray[np.uint16] | NDArray[np.uint8]
)
ImSequence: TypeAlias = (
    NDArray[np.float32] | NDArray[np.int32] | NDArray[np.uint16] | NDArray[np.uint8]
)
