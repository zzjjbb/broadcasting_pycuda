# Enable broadcasting for PyCUDA GPUArray

## Features

* extend pycuda.elementwise.Elementwise class to enable broadcasting
* extend pycuda.gpuarray.GPUArray to use the modified Elementwise

## Restrictions

* The first parameter should have the full (broadcasted) shape
  (This is not checked. If it has a smaller size, only a part will be evaluated)

* Parameters which don't have the full shape should be read-only (This is natural, right?)

## How it works

You can find this example in [test/test_broadcasting.py](test/test_broadcasting.py#L37)

Import `ElementwiseKernel` from [src](src) and use like this

```python3
ElementwiseKernel(
    "double *out, double *a, double *b, double *c",
    "out[i] = sin(a[i] * b[i]) + sqrt(c[i]) + cos(a[i]) / log(cos(b[i]) + 2)",
    broadcasting=True  # or False
)
```

Actual kernel function when `broadcasting=False`
(The `broadcasting` is `False` by default, which should be the same as original class):

```c
__global__ void kernel(double *out, double *a, double *b, double *c, unsigned long long n)
{
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;
    
    for (i = cta_start + tid; i < n; i += total_threads)
    {
        out[i] = sin(a[i] * b[i]) + sqrt(c[i]) + cos(a[i]) / log(cos(b[i]) + 2);
    }
}
```

Actual kernel function when `broadcasting=True` and called with argument
`a.shape == (3, 1, 7, 3, 10)`, `b.shape == (5, 3, 2, 7, 3, 1)`, `c.shape == (7, 3, 10)`:

```C
__global__ void kernel(double *out, double *a, double *b, double *c, unsigned long long n)
{
    unsigned tid = threadIdx.x;
    unsigned total_threads = gridDim.x*blockDim.x;
    unsigned cta_start = blockDim.x*blockIdx.x;
    unsigned i;
    unsigned __a_i, __b_i, __c_i;
    
    for (i = cta_start + tid; i < n; i += total_threads)
    {
        __a_i = i % 1260;
        __a_i = __a_i % 210 + __a_i / 420 * 210;
        __b_i = i / 10;
        __c_i = i % 210;
        out[i] = sin(a[__a_i] * b[__b_i]) + sqrt(c[__c_i]) + cos(a[__a_i]) / log(cos(b[__b_i]) + 2);
    }
}
```

## Reference

* [inducer/pycuda](https://github.com/inducer/pycuda)
