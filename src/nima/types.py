from typing import NewType, TypeVar

import numpy as np
from numpy.typing import NDArray

ImArray = TypeVar("ImArray", NDArray[np.float_], NDArray[np.int_])
ImMask = NewType("ImMask", NDArray[np.bool_])
