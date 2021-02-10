import numpy as np
from pycuda import gpuarray
from .modified_elementwise import ElementwiseKernel
from pycuda.tools import dtype_to_ctype
from pycuda.tools import context_dependent_memoize


class GPUArray(gpuarray.GPUArray):
    def __add__(self, other, sub=False, rsub=False):
        """Add an array with an array or an array with a scalar."""
        if isinstance(other, gpuarray.GPUArray):
            # add another vector
            result = type(self)(np.broadcast_shapes(self.shape, other.shape),
                                gpuarray._get_common_dtype(self, other))
            return self._axpbyz(-1 if rsub else 1, other, -1 if sub else 1, result)
        else:
            if sub:
                return super().__sub__(other)
            elif rsub:
                return super().__rsub__(other)
            else:
                return super().__add__(other)

    def __sub__(self, other):
        return self.__add__(other, sub=True)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__add__(other, rsub=True)

    def __iadd__(self, other, sub=False):
        if isinstance(other, gpuarray.GPUArray):
            return self._axpbyz(1, other, -1 if sub else 1, self)
        else:
            return super().__isub__(other) if sub else super().__iadd__(other)

    def __isub__(self, other):
        return self.__iadd__(other, sub=True)

    def __mul__(self, other):
        if isinstance(other, gpuarray.GPUArray):
            result = type(self)(np.broadcast_shapes(self.shape, other.shape),
                                gpuarray._get_common_dtype(self, other))
            return self._elwise_multiply(other, result)
        else:
            return super().__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, gpuarray.GPUArray):
            return self._elwise_multiply(other, self)
        else:
            return self.__imul__(other)

    def _axpbyz(self, selffac, other, otherfac, out, add_timer=None, stream=None):
        """Compute ``out = selffac * self + otherfac*other``,
                where `other` is a vector.."""
        if not self.flags.forc or not other.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                               "be used as arguments to this operation")
        func = get_zaxpby_kernel(self.dtype, other.dtype, out.dtype)
        func(out, selffac, self, otherfac, other)
        return out

    def _elwise_multiply(self, other, out, stream=None):
        if not self.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                               "be used as arguments to this operation")
        func = get_binary_op_kernel(self.dtype, other.dtype,
                                    out.dtype, "*")
        func(out, self, other)
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


@context_dependent_memoize
def get_binary_op_kernel(dtype_x, dtype_y, dtype_z, operator):
    return ElementwiseKernel(
        "%(tp_z)s *z, %(tp_x)s *x, %(tp_y)s *y" % {
            "tp_x": dtype_to_ctype(dtype_x),
            "tp_y": dtype_to_ctype(dtype_y),
            "tp_z": dtype_to_ctype(dtype_z),
        },
        "z[i] = x[i] %s y[i]" % operator,
        "multiply", broadcasting=True)
