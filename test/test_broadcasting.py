from unittest import TestCase
import os
import platform
import warnings
import pycuda.compiler

if platform.system() == 'Windows':
    # eliminate "non-UTF8 char" warnings
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']
    # remove code below if you have valid C compiler in `PATH` already
    import glob

    CL_PATH = max(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio"
                            r"\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
    os.environ['PATH'] += ";" + CL_PATH[:-7]

from itertools import chain
from pycuda import gpuarray
import numpy as np
from src import GPUArray, ElementwiseKernel
import pycuda.autoinit


class TestBroadcasting(TestCase):
    float_dtype = (np.float64, np.float32)

    def setUp(self) -> None:
        self.rng = np.random.default_rng()

    def assertArrayEqual(self, a, b, delta=0.):
        # currently don't want to directly calculate GPUArray with numpy function
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        self.assertEqual(a.shape, b.shape)
        self.assertLessEqual(np.linalg.norm(a - b), delta)

    def test_ElementwiseKernel(self):
        # o = broadcast(a=[3, 1, 2, 3, 4], b=[2, 3, 2, 2, 3, 1], expected_output=(3, 1, 2, 3, 4))
        a_cpu = self.rng.random((3, 1, 7, 3, 10), dtype=np.float64)
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_cpu = self.rng.random((5, 3, 2, 7, 3, 1), dtype=np.float64)
        b_gpu = gpuarray.to_gpu(b_cpu)
        c_cpu = self.rng.random((7, 3, 10), dtype=np.float64)
        c_gpu = gpuarray.to_gpu(c_cpu)
        out_cpu = np.sin(a_cpu * b_cpu) + np.sqrt(c_cpu) + np.cos(a_cpu) / np.log(np.cos(b_cpu) + 2)
        out_gpu = gpuarray.GPUArray(out_cpu.shape, out_cpu.dtype)
        func_kernel = ElementwiseKernel(
            "double *out, double *a, double *b, double *c",
            "out[i] = sin(a[i] * b[i]) + sqrt(c[i]) + cos(a[i]) / log(cos(b[i]) + 2)",
            broadcasting=True
        )

        func_kernel(out_gpu, a_gpu, b_gpu, c_gpu)
        self.assertArrayEqual(out_cpu, out_gpu.get(), delta=1e-10)

    def _test_GPUArray_one_type(self, dtype, cmplx=False):
        # random data with different shape but can broadcast
        a_cpu = self.rng.random((3, 1, 3, 13), dtype=dtype)
        b_cpu = self.rng.random((11, 3, 2, 3, 1), dtype=dtype)
        if cmplx:
            a_cpu = a_cpu + self.rng.random((3, 1, 3, 13), dtype=dtype) * 1j
            b_cpu = b_cpu + self.rng.random((11, 3, 2, 3, 1), dtype=dtype) * 1j

        # copy to GPU memory
        a_gpu = GPUArray(a_cpu.shape, a_cpu.dtype)
        a_gpu.set(a_cpu)
        b_gpu = GPUArray(b_cpu.shape, b_cpu.dtype)
        b_gpu.set(b_cpu)

        # also test compatibility with not modified GPUArray
        bb_gpu = gpuarray.to_gpu(b_cpu)
        self.assertNotIsInstance(bb_gpu, GPUArray)

        # normally add and sub should equal exactly
        self.assertArrayEqual(a_cpu + b_cpu, (a_gpu + b_gpu).get())
        self.assertArrayEqual(a_cpu + b_cpu, (a_gpu + bb_gpu).get())
        self.assertArrayEqual(b_cpu + a_cpu, (bb_gpu + a_gpu).get())
        self.assertArrayEqual(a_cpu + 1, (a_gpu + 1).get())
        self.assertArrayEqual(1 + a_cpu, (1 + a_gpu).get())

        self.assertArrayEqual(a_cpu - b_cpu, (a_gpu - b_gpu).get())
        self.assertArrayEqual(a_cpu - b_cpu, (a_gpu - bb_gpu).get())
        self.assertArrayEqual(b_cpu - a_cpu, (bb_gpu - a_gpu).get())
        self.assertArrayEqual(a_cpu - 1, (a_gpu - 1).get())
        self.assertArrayEqual(1 - a_cpu, (1 - a_gpu).get())

        # mul in pycuda.hpp is a little different from numpy
        self.assertArrayEqual(a_cpu * b_cpu, (a_gpu * b_gpu).get(), delta=1e-4)
        self.assertArrayEqual(a_cpu * b_cpu, (a_gpu * bb_gpu).get(), delta=1e-4)
        self.assertArrayEqual(a_cpu * b_cpu, (bb_gpu * a_gpu).get(), delta=1e-4)
        self.assertArrayEqual(a_cpu * 2, (a_gpu * 2).get(), delta=1e-4)
        self.assertArrayEqual(2 * a_cpu, (2 * a_gpu).get(), delta=1e-4)

        # in-place test
        out_gpu = a_gpu + b_gpu  # lazy way to get a broadcasted container
        out_ptr = out_gpu.ptr
        out_gpu += a_gpu
        out_gpu += 1
        out_cpu = a_cpu + b_cpu + a_cpu + 1
        self.assertArrayEqual(out_cpu, out_gpu.get())
        out_gpu -= a_gpu * 2
        out_gpu -= 1
        self.assertArrayEqual(out_cpu - a_cpu * 2 - 1, out_gpu.get())
        self.assertEqual(out_ptr, out_gpu.ptr)

    def test_GPUArray_float(self):
        for t in self.float_dtype:
            self._test_GPUArray_one_type(t)

    def test_GPUArray_complex(self):
        for t in self.float_dtype:
            self._test_GPUArray_one_type(t, cmplx=True)
