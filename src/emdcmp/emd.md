---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: py:percent,md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python (emdcmp-dev)
  language: python
  name: emdcmp-dev
---

+++ {"tags": ["remove-cell"], "editable": true, "slideshow": {"slide_type": ""}}

---
math:
  '\RR'    : '\mathbb{R}'
  '\nN'    : '\mathcal{N}'
  '\D'     : '\mathcal{D}'
  '\l'     : 'l'
  '\Me'    : '\mathcal{M}^ε'
  '\Unif'  : '\mathop{\mathrm{Unif}}'
  '\Philt' : '\widetilde{Φ}_{|#1}'
  '\EMD'   : '\mathrm{EMD}'
  '\Bemd'  : 'B_{#1}^{\mathrm{EMD}}'
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(supp_emd-implementation)=
# EMD implementation

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
import logging
import multiprocessing as mp
from collections.abc import Callable, Mapping
from math import sqrt, ceil, isclose
from itertools import product  # Only used in calibration_plot
from functools import partial
from more_itertools import all_equal
import numpy as np
from numpy.random import default_rng
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from scipy.special import erf
from tqdm.auto import tqdm

from typing import Optional, Union, Any, Literal, Tuple, List, Dict, NamedTuple
from scityping.numpy import Array

from emdcmp import Config
from emdcmp.path_sampling import generate_quantile_paths
from emdcmp.memoize import memoize

config = Config()
logger = logging.getLogger(__name__)
```

```{code-cell}
__all__ = ["interp1d", "make_empirical_risk_ppf", "draw_R_samples", "Bemd"]
```

+++ {"tags": ["remove-cell"], "editable": true, "slideshow": {"slide_type": ""}}

Notebook only imports

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
from scipy import stats
import holoviews as hv
hv.extension(config.viz.backend)
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.ERROR)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
colors = config.viz.matplotlib.colors["medium-contrast"]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(supp_emd-implementation_example-sampling-paths)=
## Path sampler test

$$\begin{aligned}
\tilde{\l} &= \log Φ\,, & Φ &\in [0, 1]\\
\tilde{σ} &= c \sin π Φ \,, & c &\in \mathbb{R}_+
\end{aligned}$$

The upper part of the yellow region is never sampled, because monotonicity prevents paths from exceeding $\log 1$ at any point. The constant $c$ is determined by a calibration experiment, and controls the variability of paths. Here we use $c=1$.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb]
---
res = 7
#Φarr = np.arange(1, 2**res) / 2**res
qstar = np.log
σtilde = lambda Φ: np.sin(Φ * np.pi)
M = 20

Φarr = np.arange(1, 2**res) / 2**res
lcurve = hv.Curve(zip(Φarr, qstar(Φarr)), kdims=["Φ"], vdims=["l"], label=r"$\langle\tilde{l}\rangle$")
σarea = hv.Area((Φarr, qstar(Φarr) - σtilde(Φarr), qstar(Φarr) + σtilde(Φarr)),
                kdims=["Φ"], vdims=["l-σ", "l+σ"])
GP_fig = σarea.opts(color="none", edgecolor="none", facecolor="#EEEEBB", backend="matplotlib") * lcurve

qhat_gen = generate_quantile_paths(qstar, σtilde, c=1, M=M, res=res, Phistart=Φarr[0])
random_colors = default_rng().uniform(0.65, 0.85, M).reshape(-1,1) * np.ones(3)  # Random light grey for each curve
qhat_curves = [hv.Curve(zip(Φhat, qhat), kdims=["Φ"], vdims=["l"], label=r"$\hat{l}$")
               .opts(color=color, backend=config.viz.backend)               
               for (Φhat, qhat), color in zip(qhat_gen, random_colors)]

Φ_fig = hv.Overlay(qhat_curves)

GP_fig * Φ_fig
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Similar test, but we allow variability in the end point. Note that now samples samples can cover all of the yellow region.

$$\begin{aligned}
\tilde{\l} &= \log Φ\,, & Φ &\in [0, 1]\\
\tilde{σ} &= c \sin \frac{3 π Φ}{4} \,,  & c &\in \mathbb{R}_+ \,.
\end{aligned}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$\begin{aligned}
\tilde{\l} &= \log Φ\,, & Φ &\in [0, 1]
\end{aligned}$$

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
res = 7
qstar = np.log
σtilde = lambda Φ: np.sin(Φ * 0.75*np.pi)
M = 20

Φarr = np.arange(1, 2**res) / 2**res
lcurve = hv.Curve(zip(Φarr, qstar(Φarr)), kdims=["Φ"], vdims=["l"], label=r"$\langle\tilde{l}\rangle$")
σarea = hv.Area((Φarr, qstar(Φarr) - σtilde(Φarr), qstar(Φarr) + σtilde(Φarr)),
                kdims=["Φ"], vdims=["l-σ", "l+σ"])
GP_fig = σarea.opts(color="none", edgecolor="none", facecolor="#EEEEBB", backend="matplotlib") * lcurve

qhat_gen = generate_quantile_paths(qstar, σtilde, c=1, M=M, res=res, Phistart=Φarr[0])
random_colors = default_rng().uniform(0.65, 0.85, M).reshape(-1,1) * np.ones(3)  # Random light grey for each curve
qhat_curves = [hv.Curve(zip(Φarr, qhat), kdims=["Φ"], vdims=["l"], label=r"$\hat{l}$")
               .opts(color=color, backend=config.viz.backend)
               for (Φhat, qhat), color in zip(qhat_gen, random_colors)]

Φ_fig = hv.Overlay(qhat_curves)

GP_fig * Φ_fig
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Serializable PPF functions

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Serializable 1d interpolator

Our [workflow tasks](./tasks.md) require arguments to be serializable, which include the callable arguments used to compute the center $q^*$ and the metric variance ($c (δ^{\EMD})^2$). These functions are almost always going to be obtained by constructing an interpolator from empirical samples, and the obvious choice for that is *SciPy*’s `interp1d`. However this type is of course not serializable out of the box.

To add serializability, we define a custom version of `interp1d` which adds the necessary functions. In addition, by reusing the name `interp1d`, *scityping* automatically makes the original class in `scipy.interpolate` also serializable. (See [*Defining serializers for preexisting types*](https://scityping.readthedocs.io/en/stable/defining-serializers.html) from the *scityping* documentation.) This way the serialization functionality is completely transparent, and users can use the standard `interp1d` function *SciPy* in their code.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
from typing import Literal
from dataclasses import dataclass
import scipy.interpolate
from scityping import Serializable
from scityping.numpy import Array

class interp1d(Serializable, scipy.interpolate.interp1d):
    """
    Subclass of `scipy.interpolate.interp1d` which is serializable within the
    `scitying` framework. Otherwise completely equivalent to the version in *scipy*.
    """
    @dataclass
    class Data:
        x: Array         # These are all possible arguments to interp1d
        y: Array         # as of scipy v1.11.1
        kind: str|int
        axis: int|None
        copy: bool|None
        bounds_error: bool|None
        fill_value: Literal["extrapolate"]|Tuple[Array, Array]|Array
        
        def encode(f):
            return (f.x, f.y, f._kind, f.axis, f.copy, f.bounds_error, f.fill_value)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Estimate the risk quantile function

Also known as the point probability function (PPF), this is the function $q(Φ)$ required by `draw_R_samples` to draw samples of the expected risk. Note that this is the quantile function *of the loss*, not of models predictions, so it is always a 1d function.

- The abscissa of the quantile function (PPF) is the cumulative probability $Φ \in [0, 1]$.
  The ordinate is the *loss* of a data sample.
- The integral of $q(Φ)$ is the expected risk: $\int_0^1 q(Φ) dΦ = R$.  
  (This is easy to see in 1d, where $dΦ = p(x)dx$.)
- The EMD approximation defines a distribution over quantile functions of the loss.
  - It sets the square root of the *metric variance* at $Φ$ to be proportional to the discrepancy between $q^*(Φ)$ and $\tilde{q}(Φ)$. We name this discrepancy $δ^{\EMD}(Φ)$.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In some cases it may be possible to derive $q$ analytically from a models equations, but in general we need to estimate it from samples. Concretelly all this does is construct a `scipy.interpolate.interp1d` object: if $L$ risk samples are given, they are sorted and assigned the cumulative probabilities $Φ = \frac{1}{L+1}, \frac{2}{L+1}, \dotsc, \frac{L}{L+1}$. Intermediate values are obtained by linear interpolation. We don’t assign  the $Φ=0$ and $Φ=1$, since it is more conservative to assume that the absolute extrema have not been sampled – instead we use linear extrapolation for the small intervals $\bigl[0, \frac{1}{L+1}\bigr)$ and $\bigl(\frac{L}{L+1}, 1\bigr]$.

If users want to use different assumptions – for example if users know that the highest possible risk is part of the sample set – then they may construct the `scipy.interpolate.interp1d` object directly.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
def make_empirical_risk_ppf(risk_samples: Array[float,1]) -> interp1d:
    """Convert a set of samples into a callable and serializable PPF function.

    The PPF describes the distribution from which the samples were drawn.
    Sample sizes of 1000-4000 are usually sufficient, although this depends on
    the shape of the PPF. As a general rule, the larger the maximum derivative
    along the PPF, the more samples are needed to resolve it accurately.
    Consequently, the best test to check whether the empirical PPF is a good
    approximation is simply to plot it and inspect the result visually.

    Concretely this just a wrapper for `scipy.interpolate.interp1d`, specialized
    to the case of interpolating a PPF. The more descriptive function and argument
    names help make higher-level code more readable.
    Instead of using this function, it is also possible to call `interp1d` directly;
    one simply needs to then also specify appropriate abscissa values.

    .. Important:: This expects to receive samples of the risk (not the model).
       Typically this is obtain with something like ``risk(model(n_data_points))``,
       where ``model`` is a generative model and ``risk`` the per-data-point risk
       function.

    .. Note:: When calling `Bemd` directly, any callable will work for the PPF
       argument. However, in order to use the `Calibrate` task under `emdcmp.tasks`,
       it is necessary for the PPF callable to be *serializable*. This package
       adds special support to make scipy’s `interp1d` class serializable, but other
       interpolators will not work out of the box with `Calibrate`.
       The easiest way to make an arbitrary callable serializable is probably to
       use a `dataclasses.dataclass`: use class attributes for the parameters,
       and define the PPF in the __call__ method. (Scityping has built-in support
       for dataclasses with simple data types.)

    """
    risk_samples = np.asarray(risk_samples)
    if risk_samples.ndim != 1:
        raise ValueError("A risk PPF should always be 1d, but the provided "
                         f"array `risk_samples` is {risk_samples.ndim}d.")
    L = len(risk_samples)
    Φarr = np.arange(1, L+1) / (L+1)
    return interp1d(Φarr, np.sort(risk_samples), fill_value="extrapolate")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Draw samples of the expected risk $R$

**Given**

- $\D^L$: A data set of $L$ samples.
- $\l_{A;\Me}: X \times Y \to \RR$: Function returning the log likelihood of a sample.
- $c$: Proportionality constant between the EMD and the metric variance when sampling increments.
- $r$: Resolution of the $\Philt{\Me,A}$ paths. Paths will discretized into $2^r$ steps. (And therefore contain $2^r + 1$ points.)
- $M$: The number of paths over which to average.

**Return**

- Array of $R$ values, drawn from from the EMD distribution

+++ {"editable": true, "slideshow": {"slide_type": ""}}

:::{note}  
:class: margin
The rule for computing `new_M` comes from the following ($ε$: `stderr`, $ε_t$: `stderr_tol`, $M'$: `new_M`)
```{math}
\begin{aligned}
ε &= σ/\sqrt{M} \\
ε' &= σ/\sqrt{M'} \\
\frac{ε^2}{ε'^2} &= \frac{M'}{M}
\end{aligned}
```
:::

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
@memoize(ignore=["path_progbar"])
def draw_R_samples(mixed_risk_ppf: Callable,
                   synth_risk_ppf: Callable,
                   c: float, *,
                   res: int=8, M: int=128, max_M: int=1024,
                   relstderr_tol: float=2**-5,  # 2⁻⁵ ≈ 0.03
                   path_progbar: Union[Literal["auto"],None,tqdm,mp.queues.Queue]=None,
                   print_relstderr: bool=False
                  ) -> Array[float, 1]:
    """
    Draw samples of the expected risk `R` using the EMD hierarchical beta process
    to generate model cumulative distribution functions. This is meant to approximate
    the expected risk on unseen data, while accounting for uncertainty in the model
    itself.
    The more accurate the model (as measured by its statistical correspondence with
    the data), the more tightly the R samples are distributed.

    Concretely this is done by comparing two quantile functions (also known as
    point probability functions):

    - The *mixed PPF* is obtained by evaluating the risk of the actual observed
      data samples.
    - The *synthetic PPF* is obtained by evaluating the risk on synthetic samples,
      generated using the same theoretical model used to compute the risk.

    Thus if the theory perfectly describes the data, then the two quantiles should
    be identical (up to sampling errors). Note that these are the quantile functions
    *of the risk*, not of models predictions, so they are always a 1d functions.

    In some sense this function boils down to calling `path_sampling.generate_quantile_paths`
    `M` times, integrating each resulting path with Simpson’s rule to get its
    expected risk `R`, and returning the resulting `R` values as an array.
    However, it also provides the very useful convenience of automatically adjusting
    the number `M` of realizations needed to estimate the mean value of `R`.
    It does this by first generating `M` realizations and then computing the
    standard error on the mean of `R`: if this is greater than `relstderr * mean(R)`,
    then new realizations are generated to try to achieve the target accuracy.
    (The number of realizations added depends on how far away we are from that target.)
    This process is repeated until either the target accuracy is reached, or the
    number of realizations reaches `max_M`. In the latter case a warning is printed.
    
    .. Note:: When using multiprocessing to call this function multiple times,
       use either a `multiprocessing.Queue` or `None` for the `progbar` argument.
    
    Parameters
    ----------
    mixed_risk_ppf: Quantile function of the risk using *observed* data samples.
    synth_risk_ppf: Quantile function of the risk using *synthetic* data samples.
    c: Proportionality constant between EMD and path sampling variance.
    res: Controls the resolution of the random quantile paths generated to compute statistics.
       Paths have ``2**res`` segments; typical values of `res` are 6, 7 and 8, corresponding
       to paths of length 64, 128 and 256. Smaller may be useful to accelerate debugging.
       Larger values may be needed in cases where the PPFs are unusually sharp.
    M: The minimum number of paths over which to average.
       Actual number may be more, to achieve the specified standard error.
    max_M: The maximum number of paths over which to average.
       This serves to prevent runaway computation in case the specified
       standard error is too low.
    relstderr_tol: The maximum relative standard error on the moments we want to allow.
       (i.e. ``stderr / |mean(R)|``). If this is exceeded after taking `M` path samples,
       the number of path samples is increased until we are under tolerance, or we have
       drawn 1000 samples. A warning is displayed if 1000 paths does not achieve tolerance.
    path_progbar: Control whether to create progress bar or use an existing one.
       - With the default value 'auto', a new tqdm progress is created.
         This is convenient, however it can lead to many bars being created &
         destroyed if this function is called within a loop.
       - To prevent this, a tqdm progress bar can be created externally (e.g. with
         ``tqdm(desc="Generating paths")``) and passed as argument.
         Its counter will be reset to zero, and its set total to `M` + `previous_M`.
       - (Multiprocessing): To support updating progress bars within child processes,
         a `multiprocessing.Queue` object can be passed, in which case no
         progress bar is created or updated. Instead, each time a quantile path
         is sampled, a value is added to the queue with ``put``. This way, the
         parent process can update a progress by consuming the queue; e.g.
         ``while not q.empty(): progbar.update()``.
         The value added to the queue is `M`+`previous_M`, which can be
         used to update the total value of the progress bar.
       - A value of `None` prevents displaying any progress bar.
    print_relstderr: Debug option. Setting to true will cause the number of realizations
       and associated standard error to be printed each time `M` is increased.
       Useful for checking if the requested resolution is reasonable.
       
    Returns
    -------
    array of floats
    """

    δemd = lambda Φ: abs(mixed_risk_ppf(Φ) - synth_risk_ppf(Φ))
                      
    # Compute m1 for enough sample paths to reach relstderr_tol
    m1 = []
    def generate_paths(M, previous_M=0, qstar=mixed_risk_ppf, δemd=δemd, c=c, res=res, progbar=path_progbar):
        for Φhat, qhat in generate_quantile_paths(qstar, δemd, c=c, M=M, res=res,
                                                  progbar=progbar, previous_M=previous_M):
            m1.append(simpson(y=qhat, x=Φhat))  # Generated paths always have an odd number of steps, which is good for Simpson's rule

    generate_paths(M)
    μ1 = np.mean(m1)
    Σ1 = np.var(m1)
    relstderr = sqrt(Σ1) / max(abs(μ1), 1e-8) / sqrt(M)  # TODO?: Allow setting abs tol instead of hardcoding 1e-8 ?
    if print_relstderr:
        print(f"{M=}, {relstderr=}")
    while relstderr > relstderr_tol and M < max_M:
        # With small M, we don’t want to put too much trust in the
        # initial estimate of relstderr. So we cap increases to doubling M.
        new_M = min(ceil( (relstderr/relstderr_tol)**2 * M ), 2*M)
        logger.debug(f"Increased number of sampled paths (M) to {new_M}. "
                     f"Previous rel std err: {relstderr}")
        if new_M > max_M:
            new_M = max_M
            logger.warning(f"Capped the number of sample paths to {max_M} "
                           "to avoid undue computation time.")
        if new_M == M:
            # Can happen due to rounding or because we set `new_M` to `max_M`
            break
        generate_paths(new_M - M, M)
        M = new_M
        μ1 = np.mean(m1)
        Σ1 = np.var(m1)
        relstderr = sqrt(Σ1) / max(abs(μ1), 1e-8) / sqrt(M)  # TODO?: Allow setting abs tol instead of hardcoding 1e-8 ?
        if print_relstderr:
            print(f"{M=}, {relstderr=}")
        
    if relstderr > relstderr_tol:
        logger.warning("Requested std err tolerance was not achieved. "
                       f"std err: {relstderr}\nRequested max std err: {relstderr_tol}")
    return np.array(m1)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(sec_emd_test-sampling)=
### Test sampling of expected risk $R$

$$\begin{aligned}
x &\sim \Unif(0, 3) \\
y &\sim e^{-λx} + ξ
\end{aligned}$$

:::{list-table}

* - Theory model $A$:
  - $λ=1$
  - $ξ \sim \nN(0, 1)$.
* - Theory model $B$:
  - $λ=1.1$
  - $ξ \sim \nN(0, 1)$.
* - True data-generating model:
  - $λ=1$
  - $ξ \sim \nN(-0.03, 1)$. 
 
:::

In this example, neither model $A$ nor $B$ is a perfect fit to the data, since they both incorrectly assume an unbiased noise. Moreover, both models seem to predict the observations equally well; in other words, we expect the EMD criterion to be *equivocal* between models $A$ and $B$.

Within the EMD framework, models are compared as usual based on their expected risk $R$. This captures aleatoric uncertainty – i.e. randomness inherent to the model, such as the $ξ$ random variable above. The EMD criterion then further captures *epistemic* uncertainty by treating $R$ itself as a random variable, and considering *its* distribution. Roughly speaking, the better a model is at predicting the data distribution, the tighter its $R$ distribution will be. (For example, a model can have a lot of noise, but if we can predict the statistics of that noise accurately, then the distribution on $R$ will be tight and its uncertainty low.)

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
λ = 1
λB = 1.2
σ = 0.25
δy = -0.03
L = 400
seed = 123

def generative_data_model(L, rng=None):
    rng = default_rng(rng)
    x = rng.uniform(0, 3, L)
    y = np.exp(-λ*x) + rng.normal(-0.03, σ, L)
    return x, y

def generative_theory_modelA(L, rng=None):
    rng = default_rng(rng)
    x = rng.uniform(0, 3, L)
    y = np.exp(-λ*x) + rng.normal(0, σ, L)
    return x, y
def riskA(xy):
    "Negative log likelihood of model A"
    x, y = xy
    return -stats.norm(0, 1).logpdf(y - np.exp(-λ*x))  # z = exp(-λ*x)

def generative_theory_modelB(L, rng=None):
    rng = default_rng(rng)
    x = rng.uniform(0, 3, L)
    y = np.exp(-λB*x) + rng.normal(0, σ, L)
    return x, y
def riskB(xy):
    "Negative log likelihood of model B"
    x, y = xy
    return -stats.norm(0, 1).logpdf(y - np.exp(-λB*x))  # z = exp(-λ*x)

observed_data = generative_data_model(L, seed)
synth_dataA    = generative_theory_modelA(L, seed*2)  # Use different seeds for different models
synth_dataB    = generative_theory_modelB(L, seed*3)

mixed_ppfA = make_empirical_risk_ppf(riskA(observed_data))
synth_ppfA = make_empirical_risk_ppf(riskA(synth_dataA))

mixed_ppfB = make_empirical_risk_ppf(riskB(observed_data))
synth_ppfB = make_empirical_risk_ppf(riskB(synth_dataB))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In this example, the discrepancy between the theoretical models and the observed data distribution is very small, so the differences between quantile curves is similarly very small.

- **Synthetic** PPF — Same theoretical model for both the defining the risk and generating the (synthetic) data.
- **Mixed** PPF – Different models for the risk and data: Again a theoretical model is used to define the risk, but now it is evaluated on real observed data.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
# Data panel
xarr = np.linspace(0, 3)
scat_data = hv.Scatter(zip(*observed_data), group="data", label="data")
curve_A = hv.Curve(zip(xarr, np.exp(-λ*xarr)), group="model A", label="model A")
curve_B = hv.Curve(zip(xarr, np.exp(-λB*xarr)), group="model B", label="model B")
panel_data = scat_data * curve_A * curve_B

panel_data.opts(
    hv.opts.Scatter("data", color="grey", alpha=0.5), hv.opts.Scatter("data", s=6, backend="matplotlib"),
    hv.opts.Curve("model A", color=colors["dark blue"], alpha=0.7),
    hv.opts.Curve("model B", color=colors["dark red"], alpha=0.7)
)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input, active-ipynb]
---
# Quantile function panels
Φarr = np.linspace(0, 1, 100)
curve_synthA = hv.Curve(zip(Φarr, synth_ppfA(Φarr)), kdims=["Φ"], vdims=["q"], group="synth", label="model A")
curve_synthB = hv.Curve(zip(Φarr, synth_ppfB(Φarr)), kdims=["Φ"], vdims=["q"], group="synth", label="model B")
curve_mixedA = hv.Curve(zip(Φarr, mixed_ppfA(Φarr)), kdims=["Φ"], vdims=["q"], group="mixed", label="model A")
curve_mixedB = hv.Curve(zip(Φarr, mixed_ppfB(Φarr)), kdims=["Φ"], vdims=["q"], group="mixed", label="model B")
fig = curve_synthA * curve_synthB * curve_mixedA * curve_mixedB

zoomrect = hv.Rectangles([(.6, .95, .99, 1.15,)]).opts(facecolor="none", edgecolor="grey")
legendtext = hv.Text(0.61, 1.05, "Dashed lines:\nmixed PPF", halign="left")
layout = panel_data + fig*zoomrect + fig.redim.range(Φ=(0.6, .99), q=(.95, 1.15))*legendtext

layout.opts(
    hv.opts.Curve(alpha=0.8, axiswise=True),
    hv.opts.Curve("synth.model A", color=colors["dark blue"]), hv.opts.Curve("mixed.model A", color=colors["dark blue"]),
    hv.opts.Curve("synth.model B", color=colors["light red"]), hv.opts.Curve("mixed.model B", color=colors["light red"]),
    hv.opts.Curve("synth", linestyle="solid", backend="matplotlib"), hv.opts.Curve("synth", line_dash="solid", backend="bokeh"),
    hv.opts.Curve("mixed", linestyle="dashed", backend="matplotlib"), hv.opts.Curve("mixed", line_dash="dashed", backend="bokeh"),
    hv.opts.Curve(aspect=1.3, backend="matplotlib"),
    hv.opts.Layout(fig_inches=4, backend="matplotlib"),
    hv.opts.Layout(sublabel_format="")
)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
# R dist panels
RA_lst = draw_R_samples(mixed_ppfA, synth_ppfA, c=1, M=100, print_relstderr=True)
RB_lst = draw_R_samples(mixed_ppfB, synth_ppfB, c=1, M=100, print_relstderr=True)

RAlines = hv.Spikes(RA_lst, kdims=["R"], group="model A").opts(spike_length=50)
RBlines = hv.Spikes(RB_lst, kdims=["R"], group="model B").opts(spike_length=50)

distA = hv.Distribution(RA_lst, kdims=["$R$"], vdims=["$p(R)$"], group="model A")
distB = hv.Distribution(RB_lst, kdims=["$R$"], vdims=["$p(R)$"], group="model B")

panel_RA = distA * RAlines
panel_RB = distB * RBlines
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Each model induces a distribution for its expected risk, so with two models $A$ and $B$ we have distributions for $R_A$ and $R_B$.
The figures below show the sampled values for $R_A$ and $R_B$, overlayed with a kernel density estimate of their distribution.

In this case both distributions are very tight, and any difference between them are due as much to finite sampling than to the likelihood picking up which one is the better fit. This translates into distributions with very high overlap, and therefore a probability approximately ½ that model $A$ is better than $B$. In other words, the criterion is *equivocal* between $A$ and $B$, as we expected.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, hide-input]
---
fig = panel_RA + panel_RB + panel_RA*panel_RB

panel_RA.opts(
    hv.opts.Spikes("model A", color=colors["dark blue"], alpha=0.25),
    hv.opts.Distribution("model A", color="none", edgecolor=colors["dark blue"], facecolor=colors["light blue"], backend="matplotlib")
)
panel_RB.opts(
    hv.opts.Spikes("model B", color=colors["dark red"], alpha=0.25),
    hv.opts.Distribution("model B", color="none", edgecolor=colors["dark red"], facecolor=colors["light red"], backend="matplotlib")
)

fig.opts(
    hv.opts.Overlay(aspect=1.3),
    hv.opts.Layout(fig_inches=3, backend="matplotlib"),
    hv.opts.Layout(sublabel_format="")
)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(supp_emd-implementation_Bemd)=
## Implementation of $B^{\mathrm{emd}}_{AB;c}$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The EMD criterion is defined as
```{math}
\begin{aligned}
\Bemd{AB;c} &:= P(R_A < R_B \mid c) \\
    &\approx \frac{1}{M_A M_B} \sum_{i=1}^{M_A} \sum_{j=1}^{M_B} \boldsymbol{1}_{R_{A,i} < R_{B,j}}\,,
\end{aligned}
```
where $c$ is a scale parameter and $M_A$ (resp. $M_B$) is the number of samples of $R_A$ (resp. $R_B$). The expression $\boldsymbol{1}_{R_{A,i} < R_{B,j}}$ is one when $R_{A,i} < R_{B,j}$ and zero otherwise.
We can write the sum as nested Python loops:

:::{margin}
(`RA_lst` and `RB_lst` are the expected risk samples generated in the example above.)
:::

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
s = 0
for rA in RA_lst:
    for rB in RB_lst:
        s += (rA < rB)
s / len(RA_lst) / len(RB_lst)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

However it is much faster (about 50x in this test example) to use a NumPy ufunc:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
np.less.outer(RA_lst, RB_lst).mean()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The `Bemd` function is a convenience function for comparing two models, which calls `draw_R_samples` (once for each models) and then computes the criterion value as above. The essence of the function is three lines:

```python
def Bemd(mixed_risk_ppfA: Callable, mixed_risk_ppfB: Callable,
         synth_risk_ppfA: Callable, synth_risk_ppfB: Callable,
         c: float, *, ...):
  RA_lst = draw_R_samples(mixed_risk_ppfA, synth_risk_ppfA, c=c, ...)
  RB_lst = draw_R_samples(mixed_risk_ppfB, synth_risk_ppfB, c=c, ...)
  return np.less.outer(RA_lst, RB_lst).mean()
```

The rest of the code is wraps the necessary boilerplate for dispatching the generation of $R$ samples to multiple processes, and keeping progress bars updated.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
def mp_wrapper(f: Callable, *args, out: "mp.queues.Queue", **kwargs):
    "Wrap a function by putting its return value in a Queue. Used for multiprocessing."
    out.put(f(*args, **kwargs))
    
LazyPartial = Union[Callable, Tuple[Callable, Mapping]]
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
@memoize(ignore=["progbarA", "progbarB", "use_multiprocessing"])
def Bemd(mixed_risk_ppfA: Callable, mixed_risk_ppfB: Callable,
         synth_risk_ppfA: Callable, synth_risk_ppfB: Callable,
         c: float, *,
         res: int=7, M: int=32, max_M: int=1024,
         relstderr_tol: float=2**-5,  # 2⁻⁵ ≈ 0.03
         progbarA: Union[tqdm,Literal['auto'],None]='auto',
         progbarB: Union[tqdm,Literal['auto'],None]='auto',
         use_multiprocessing: bool=True
        ) -> float:

    """
    Compute the EMD criterion for two models, whose risk distributions are described
    by a *mixed risk PPF* (mixing observed samples with theoretical risk function)
    and a *synthetic risk PPF* (from theoretical samples and theoretical risk function).

    This is accomplished by calling `draw_R_samples` for each model, and using the
    result to compute the probability that model A has lower risk than model B.

    The main benefit of this function is that it can manage a multiprocessing pool
    to perform calculations in parallel and keep progress bars updated.
    
    For more details on the parameters, see `draw_R_samples`.
    
    Parameters
    ----------
    mixed_risk_ppfA, mixed_risk_ppfB: Quantile functions of the risk using
       *observed* data samples.
    synth_risk_ppfA, synth_risk_ppfB: Quantile function of the risk using
       *synthetic* data samples.
    c: Proportionality constant between EMD and path sampling variance.
    res: Controls the resolution of the random quantile paths generated to compute statistics.
       Paths have length ``2**res + 1``; typical values of `res` are 6, 7 and 8, corresponding
       to paths of length 64, 128 and 256. Smaller may be useful to accelerate debugging.
       Larger values may be needed in cases where the PPFs are unusually sharp.
    M: The minimum number of paths over which to average.
       Actual number may be more, to achieve the specified standard error.
    max_M: The maximum number of paths over which to average.
       This serves to prevent runaway computation in case the specified
       standard error is too low.
    relstderr_tol: The maximum relative standard error on the moments we want to allow.
       (i.e. ``stderr / |μ1|``). If this is exceeded after taking `M` path samples,
       the number of path samples is increased until we are under tolerance, or we have
       drawn 1000 samples. A warning is displayed if 1000 paths does not achieve tolerance.
    progbarA, probgbarB: Control whether to create progress bar or use an existing one.
       These progress bars track the number of generated quantile paths.
       - With the default value 'auto', a new tqdm progress is created.
         This is convenient, however it can lead to many bars being created &
         destroyed if this function is called within a loop.
       - To prevent this, a tqdm progress bar can be created externally (e.g. with
         ``tqdm(desc="Generating paths")``) and passed as argument.
         Its counter will be reset to zero, and its set total to `M` + `previous_M`.
       - A value of `None` prevents displaying any progress bar.
    use_multiprocessing: If `True`, the statistics for models A and B are
       computed simultaneously; otherwise they are computed sequentially.
       Default is `True`.
       One reason not to use multiprocessing is if this call is part of a
       higher-level loop with is itself parallelized: child multiprocessing
       processes can’t spawn their own child processes.

    TODO
    ----
    - Use separate threads to update progress bars. This should minimize their
      tendency to lag behind the actual number of sampled paths.
    """


    # NB: Most of this function is just managing mp processes and progress bars
    if isinstance(progbarA, tqdm):
        close_progbarA = False  # Closing a progbar prevents it from being reused
    elif progbarA == 'auto':  # NB: This works because we already excluded tqdm (tqdm types raise AttributeError on ==)
        progbarA = tqdm(desc="sampling quantile fns (A)")
        close_progbarA = True
    else:  # With `progbarA=None`, we don’t create a progbar, so nothing to close.
        close_progbarA = False
    if isinstance(progbarB, tqdm):
        close_progbarB = False
    elif progbarB == 'auto':
        progbarB = tqdm(desc="sampling quantile fns (B)")
        close_progbarB = True
    else:
        close_progbarB = False
    
    if not use_multiprocessing:
        RA_lst = draw_R_samples(
            mixed_risk_ppfA, synth_risk_ppfA, c=c,
            res=res, M=M, max_M=max_M, relstderr_tol=relstderr_tol,
            path_progbar=progbarA)
        RB_lst = draw_R_samples(
            mixed_risk_ppfB, synth_risk_ppfB, c=c,
            res=res, M=M, max_M=max_M, relstderr_tol=relstderr_tol,
            path_progbar=progbarB)
        
    else:
        progqA = mp.Queue() if progbarA is not None else None  # We manage the progbar ourselves. The Queue is used for receiving
        progqB = mp.Queue() if progbarA is not None else None  # progress updates from the function
        outqA = mp.Queue()   # Function output values are returned via a Queue
        outqB = mp.Queue()
        _draw_R_samples_A = partial(
            draw_R_samples,
            mixed_risk_ppfA, synth_risk_ppfA, c=c,
            res=res, M=M, max_M=max_M, relstderr_tol=relstderr_tol,
            path_progbar=progqA)
        _draw_R_samples_B = partial(
            draw_R_samples,
            mixed_risk_ppfB, synth_risk_ppfB, c=c,
            res=res, M=M, max_M=max_M, relstderr_tol=relstderr_tol,
            path_progbar=progqB)
        pA = mp.Process(target=mp_wrapper, args=(_draw_R_samples_A,),
                        kwargs={'path_progbar': progqA, 'out': outqA})
        pB = mp.Process(target=mp_wrapper, args=(_draw_R_samples_B,),
                        kwargs={'path_progbar': progqB, 'out': outqB})
        pA.start()
        pB.start()
        progbar_handles = ( ([(progqA, progbarA)] if progbarA is not None else [])
                           +([(progqB, progbarB)] if progbarB is not None else []) )
        if progbar_handles:
            for _, progbar in progbar_handles:
                progbar.reset()  # We could reset the total here, but already reset it below
            while pA.is_alive() or pB.is_alive():
                for (progq, progbar) in progbar_handles:
                    if not progq.empty():
                        n = 0
                        while not progq.empty():  # Flush the progress queue, then update the progress bar.
                            total = progq.get()   # Otherwise the progress bar may not keep up
                            n += 1
                        if total != progbar.total:
                            progq.dynamic_miniters = False  # Dynamic miniters doesn’t work well when we mess around with the total
                            # Reset the max for the progress bar
                            progbar.total = total
                            if "notebook" in str(progbar.__class__.mro()):  # Specific to tqdm_notebook
                                progbar.container.children[1].max = total  
                                progbar.container.children[1].layout.width = None  # Reset width; c.f. progbar.reset()
                        progbar.update(n)

        pA.join()
        pB.join()
        pA.close()
        pB.close()
        # NB: Don't close progress bars unless we created them ourselves
        if close_progbarA: progbarA.close()
        if close_progbarB: progbarB.close()
        RA_lst = outqA.get()
        RB_lst = outqB.get()
    
    return np.less.outer(RA_lst, RB_lst).mean()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Calling `Bemd` returns the same value as above, up to sampling error:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
Bemd(mixed_ppfA, mixed_ppfB, synth_ppfA, synth_ppfB, c=1)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-input]
---
from emdcmp.utils import GitSHA
GitSHA()
```
