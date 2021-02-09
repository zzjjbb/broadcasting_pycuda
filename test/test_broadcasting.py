from unittest import TestCase
import os
import platform
import warnings
import numpy as np
import pycuda.compiler
from pycuda import gpuarray

if platform.system() == 'Windows':
    # eliminate "non-UTF8 char" warnings
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']
    # remove code below if you have valid C compiler in `PATH` already
    import glob

    CL_PATH = max(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio"
                            r"\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
    os.environ['PATH'] += ";" + CL_PATH[:-7]

from functools import lru_cache
from warnings import warn
import re
from pycuda import elementwise, gpuarray, reduction
from pycuda.tools import dtype_to_ctype, VectorArg, ScalarArg
import numpy as np
from ssnp.utils.elementwise import broadcast, ElementwiseKernel


class TestBroadcasting(TestCase):
    def setUp(self) -> None:
        pass

    def test_ElementwiseKernel(self):
        # o = broadcast(a=[3, 1, 2, 3, 4], b=[2, 3, 2, 2, 3, 1], expected_output=(3, 1, 2, 3, 4))
        a_cpu = np.random.rand(3, 1, 20, 3, 40)
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_cpu = np.random.rand(200, 3, 2, 20, 3, 1)
        b_gpu = gpuarray.to_gpu(b_cpu)
        c_cpu = np.random.rand(20, 3, 40)
        c_gpu = gpuarray.to_gpu(c_cpu)
        out_cpu = np.sin(a_cpu * b_cpu) + np.sqrt(c_cpu) + np.cos(a_cpu) / np.log(np.cos(b_cpu) + 2)
        out_gpu = gpuarray.GPUArray(out_cpu.shape, out_cpu.dtype)
        func_kernel = ElementwiseKernel(
            "double *out, double *a, double *b, double *c",
            "out[i] = sin(a[i] * b[i]) + sqrt(c[i]) + cos(a[i]) / log(cos(b[i]) + 2)",
            broadcasting=True
        )

        func_kernel(out_gpu, a_gpu, b_gpu, c_gpu)
        print(np.linalg.norm(out_cpu - out_gpu.get()))

    def test_GPUArray(self):
        a_cpu = np.random.rand(3, 1, 20, 3, 40)
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_cpu = np.random.rand(200, 3, 2, 20, 3, 1)
        b_gpu = gpuarray.to_gpu(b_cpu)
        c_cpu = np.random.rand(20, 3, 40)
        c_gpu = gpuarray.to_gpu(c_cpu)
        out_cpu = np.sin(a_cpu * b_cpu) + np.sqrt(c_cpu) + np.cos(a_cpu) / np.log(np.cos(b_cpu) + 2)
        out_gpu = gpuarray.GPUArray(out_cpu.shape, out_cpu.dtype)