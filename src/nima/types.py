"""Type definitions used throughout the clophfit package.

It defines the following types:

- ImArray: for float or int images.

- ImMask: for bool image masks.
"""

from typing import NewType, TypeVar

import numpy as np
from numpy.typing import NDArray

ImArray = TypeVar("ImArray", NDArray[np.float_], NDArray[np.int_])
ImMask = NewType("ImMask", NDArray[np.bool_])
