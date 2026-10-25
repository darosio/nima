"""Type definitions used throughout the clophfit package.

It defines the following types:

- ImArray: for float or int images.

- ImMask: for bool image masks.
"""

import numpy as np
from numpy.typing import NDArray

# or derive class ImSequence(np.ndarray):  # or extend NDArray manually if needed
# ...


type ImMask = NDArray[np.bool_]

type ImVector = (
    NDArray[np.float32] | NDArray[np.int32] | NDArray[np.uint16] | NDArray[np.uint8]
)
type ImFrame = (
    NDArray[np.float32] | NDArray[np.int32] | NDArray[np.uint16] | NDArray[np.uint8]
)
type ImSequence = (
    NDArray[np.float32] | NDArray[np.int32] | NDArray[np.uint16] | NDArray[np.uint8]
)
