## GP

### ExactGP
1. It is most simple vanilla GP regression class. It takes the following arguments:
- `kernel`: the kernel to use (default: `RBFKernel`)
- `noise`: the noise to use (default: `HomoskedasticNoise`)
- `mean`: the mean to use (default: `ConstantMean`)

2. `initialise_params(key, X, X_inducing)` method initialises the parameters of the model. `X` is passed to infer the dimensionality of X for kernel e.g. if `X` is (5, 3) and ARD is true, kernel lengthscale will be initialized of shape (3,). `X_inducing` is optional argument and it is only used when using `GibbsKernel` and/or `HeteroScedasticNoise`.

### SparseGP
1. It is a sparse GP regression class. It takes all the arguments of `ExactGP` and the following additional arguments:
- `method` - the sparse gp method to be used (default: `vfe`, other options: `dtc`, `fitc`)

2. `initialise_params(key, X, X_inducing)` method initialises the parameters of the model. `X_inducing` is a mandatory argument and it is used to initialize the inducing points.

## Kernels

1. This library implements the following common kernels: `RBFKernel`, `Matern12Kernel`, `Matern32Kernel`, `Matern52Kernel`, `LinearKernel` and a special kernel `GibbsKernel`. Apart from these, `SumKernel` and `ProductKernel` are also implemented which can be used to combine kernels.

2. All of them accept their parameter at initialization time or they will initialize parameters to default values. 

3. `initialise_params(key, X, X_inducing)` method initialises the parameters of the kernel. In some cases, `X` is needed and in other `X_inducing` but sometimes both might also be needed. In detail
- `X` is needed when initializing all kernels except `GibbsKernel`
- `X_inducing` is needed when initializing `GibbsKernel`
- Both `X` and `X_inducing` are needed when initializing a `SumKernel` or `ProductKernel` because they may contain `GibbsKernel` as one of their components.

### Kernel combinations

1. Combinations of these kernels can be done like general algebra. For example, if there are three kernels `k1`, `k2` and `k3`, then the following is one way to combine them: `k1 + (k2 * k3)`. This is equivalent to `SumKernel(k1=k1, k2=ProductKernel(k1=k2, k2=k3))`.

### Gibbs kernel

1. GibbsKernel is a special kernel which can help us learn input dependent lengthscale and variance. It takes the following arguments:
- `flex_scale`: If input dependent lengthscale is to be learned, then this should be set to `True`. Default: `True`
- `flex_variance`: If input dependent variance is to be learned, then this should be set to `True`. Default: `True`
- `X_inducing`: The inducing points to be used.

2. **If `X_inducing` is not passed then it tries to use `X_inducing` from GP parameters which is only possible if GP is `SparseGP`.**

## Noise

### HomoskedasticNoise

1. It is the most simple noise model. It takes the following arguments:
- `variance`: The variance of the noise. Default: `1.0`

### HeteroskedasticNoise

1. It is a heteroskedastic noise model. It takes the following arguments:

- `latent_log_noise`: The latent log noise to be used with size of inducing points. Default is zeros. 
- `X_inducing`: The inducing points to be used.
- `use_kernel_inducing`: If `True`, then `X_inducing` is used from the kernel which must be `GibbsKernel`. Default: `True`.

2. **If `X_inducing` is not passed then it checks `use_kernel_inducing` and if it is `True` then it tries to use `X_inducing` from the kernel which is only possible if kernel is `GibbsKernel`. If it is `False` then it tries to use `X_inducing` from GP parameters which is only possible if GP is `SparseGP`.**


## Directions for Non-stationary Heteroscedastic GP experimentation

### If you want to jointly learn `X_inducing` for both `GibbsKernel` and `HeteroscedasticNoise`
```python
kernel = GibbsKernel(X_inducing=X_inducing)
noise = HeteroscedasticNoise()
```

### If you want to separately learn `X_inducing` for both `GibbsKernel` and `HeteroscedasticNoise`
```python
kernel = GibbsKernel(X_inducing=X_inducing)
noise = HeteroscedasticNoise(X_inducing=X_inducing)
```

### If you want to learn only Heteroscedastic Noise
```python
kernel = GibbsKernel(flex_scale=False, flex_variance=False)
noise = HeteroscedasticNoise(X_inducing=X_inducing)
```

### If you want to learn heteroscedastic noise with input dependent lengthscale while jointly learning `X_inducing` for both `GibbsKernel` and `HeteroscedasticNoise`
```python
kernel = GibbsKernel(flex_variance=False, X_inducing=X_inducing)
noise = HeteroscedasticNoise()
```

### If your model is SparseGP and you want to jointly learn `X_inducing` for `SparseGP`, `GibbsKernel` and `HeteroscedasticNoise`.
```python
kernel = GibbsKernel()
noise = HeteroscedasticNoise(use_kernel_inducing=False)
gp = SparseGP(kernel=kernel, noise=noise)
params = gp.initialise_params(key, X=X, X_inducing=X_inducing)
```

### If your model is SparseGP and you want to learn `X_inducing` separately for `SparseGP` and jointly for `GibbsKernel` and `HeteroscedasticNoise`.
```python
kernel = GibbsKernel(X_inducing=X_inducing)
noise = HeteroscedasticNoise(use_kernel_inducing=True)
gp = SparseGP(kernel=kernel, noise=noise)
params = gp.initialise_params(key, X=X, X_inducing=X_inducing)
```
