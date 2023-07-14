import sys

import numpy as np
import numpy.typing as npt

Float = np.double
UFloat = float | Float | np.double

Array = npt.NDArray[Float]
ArrayMask = npt.NDArray[np.bool_]

if sys.version_info >= (3, 9):
    from typing import Annotated

    ArrayND = Annotated[Array, ("ND",)]
    ArrayNA = Annotated[Array, ("NA",)]
    ArrayNF = Annotated[Array, ("NF",)]
    ArrayNDxND = Annotated[Array, ("ND", "ND")]
    ArrayNDxNA = Annotated[Array, ("ND", "NA")]
    ArrayNDxNF = Annotated[Array, ("ND", "NF")]
    ArrayNAxNF = Annotated[Array, ("NA", "NF")]
    ArrayNDxNAxNF = Annotated[Array, ("ND", "NA", "NF")]
    ArrayMaskNA = Annotated[ArrayMask, ("NA",)]
else:
    ArrayND = Array
    ArrayNA = Array
    ArrayNF = Array
    ArrayNDxND = Array
    ArrayNDxNA = Array
    ArrayNDxNF = Array
    ArrayNAxNF = Array
    ArrayNDxNAxNF = Array
    ArrayMaskNA = ArrayMask
