import numpy as np
from pycuda import gpuarray
from .modified_elementwise import ElementwiseKernel
from pycuda.tools import dtype_to_ctype
from pycuda.tools import context_dependent_memoize


class GPUArray(gpuarray.GPUArray):
    def __add__(self, other, sub=False):
        """Add an array with an array or an array with a scalar."""
        if isinstance(other, GPUArray):
            # add another vector
            result = self.__class__(np.broadcast_shapes(self.shape, other.shape),
                                    gpuarray._get_common_dtype(self, other))
            return self._axpbyz(1, other, -1 if sub else 1, result)
        else:
            return super().__sub__(other) if sub else super().__add__(other)

    def __sub__(self, other):
        return self.__add__(other, sub=True)

    def __iadd__(self, other, sub=False):
        if isinstance(other, GPUArray):
            return self._axpbyz(1, other, -1 if sub else 1, self)
        else:
            return super().__isub__(other) if sub else super().__iadd__(other)

    def __isub__(self, other):
        return self.__iadd__(other, sub=True)

    def _axpbyz(self, selffac, other, otherfac, out, add_timer=None, stream=None):
        """Compute ``out = selffac * self + otherfac*other``,
                where `other` is a vector.."""
        if not self.flags.forc or not other.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                               "be used as arguments to this operation")

        func = get_zaxpby_kernel(self.dtype, other.dtype, out.dtype)
        func(out, selffac, self, otherfac, other)
        return out


@context_dependent_memoize
def get_zaxpby_kernel(dtype_x, dtype_y, dtype_z):
    return ElementwiseKernel(
        "%(tp_z)s *z, %(tp_x)s a, %(tp_x)s *x, %(tp_y)s b, %(tp_y)s *y" % {
            "tp_x": dtype_to_ctype(dtype_x),
            "tp_y": dtype_to_ctype(dtype_y),
            "tp_z": dtype_to_ctype(dtype_z),
        },
        "z[i] = a*x[i] + b*y[i]",
        "zaxpby", broadcasting=True)
