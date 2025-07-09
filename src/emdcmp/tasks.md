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
  '\Bepis' : 'B^{\mathrm{epis}}_{#1}'
  '\Bemd'  : 'B_{#1}^{\mathrm{EMD}}'
---

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
from __future__ import annotations
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Tasks

Running experiments via [SumatraTasks](https://sumatratask.readthedocs.io/en/latest/basics.html) has two purposes:
- Maintaining an electronic lab book: recording all input/code/output triplets, along with a bunch of metadata to ensure reproducibility (execution date, code versions, etc.)
- Avoid re-running calculations, with hashes that are both portable and long-term stable.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb]
---
from config import config   # Notebook
```

```{raw-cell}
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: ''
tags: [active-py]
---
from .config import config  # Python script
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
import abc
import psutil
import logging
import time
import multiprocessing as mp
import numpy as np
from functools import partial
from itertools import repeat
from typing import (
    TypeVar, Optional, Union, Any, Callable,
    Dict, Tuple, List, Iterable, NamedTuple, Literal)
from dataclasses import dataclass, is_dataclass, replace
from scityping import Serializable, Dataclass, Type
from tqdm.auto import tqdm
from scityping.functions import PureFunction
# Make sure Array (numpy) and RV (scipy) serializers are loaded
import scityping.numpy
import scityping.scipy

from smttask import RecordedTask, TaskOutput
from smttask.workflows import ParamColl, SeedGenerator

import emdcmp as emd
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
logger = logging.getLogger(__name__)
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
__all__ = ["Calibrate", "EpistemicDist", "CalibrateOutput"]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(code_calibration-distribution)=
## Epistemic distribution

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
@dataclass
class Experiment:
    """Reference implementation for an experiment container;
    represents one sample ω from the epistemic distribution.

    Users may use this dataclass, or define their own: Experiment containers
    need not subclass this class, but must provide the same attributes.

    One reason to use a custom experiment class is to ensure that heavy
    computations (like fitting the candidate models) happen on the child
    MP processes.
    """
    data_model: DataModel
    candidateA: CandidateModel
    candidateB: CandidateModel
    QA: RiskFunction
    QB: RiskFunction
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
@dataclass(frozen=True)
class EpistemicDist(abc.ABC):
    """Generic template for a calibration distribution:
    
    in effect a calibration distribution with no calibration parameters.
    For actual use you need to subclass `EpistemicDist` and extend it with 
    parameters relevant to your models.

    Using this class is not actually required for the `Calibrate` task: any
    frozen dataclass will do. The only requirements for the dataclass are:

    - That iterating over it yields data models.
    - That it defines `__len__`.
    - That all its parameters are serializable.
    - That it be created with ``frozen=True``.

    Users can choose to subclass this class, or just use it as a template.

    .. Note:: If subclassing, the first argument will always be `N` since
       subclasses append their parameters to the base class.
    """
    N: int|Literal[np.inf]     # Number of data models, i.e. length of iterator   
    
    @abc.abstractmethod
    def __iter__(self) -> Experiment:
        raise NotImplementedError
        # rng = <create & seed an RNG using the dist parameters as entropy>
        # for n in range(self.N):
        #     <draw calibration params using rng>
        #     yield <data model>, <candidate A>, <candidate B>, <loss A>, <loss B>

    def __len__(self):
        return self.N
    
    def generate(self, N: int):
        """Return a copy of EpistemicDist which will yield `N` models.
        
        :param:N: Number of models to return.
        """
        return replace(self, N=N)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Calibration task

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Task parameters

| Parameter | Description |
|-----------|-------------|
| `c_list` | The values of $c$ we want to test. |
| `models` | Sequence of $N$ quintuplets of models (data generation, candidate A, candidate B, loss A, loss B) drawn from a calibration distribution. Typically, but not necessarily, a subclass of `EpistemicDist`: any dataclass satisfying the requirements listed in `EpistemicDist` is accepted. |
| `Ldata` | Data set size used to construct the empirical PPF for models $A$ and $B$. Ideally commensurate with the actual data set used to assess models. |
| `Linf` | Data set size considered equivalent to "infinite". Used to compute $\Bepis{}$ |

The value of $N$ is determined from `len(models)`, so the `models` iterable should define its length.

#### Config values:

| Parameter | Description |
|-----------|-------------|
| `ncores`  |  Number of CPU cores to use. |

##### Effects on compute time

The total number of experiments will be
```{math}
N \times \lvert\mathtt{c\_list}\rvert \times \text{(# parameter set distributions)} \,.
```
In the best scenario, one can expect compute times to be 2.5 minutes / experiment.[^how-to-make-faster] So expect this to take a few hours.

Results are cached on-disk with [joblib.Memory](https://joblib.readthedocs.io/en/latest/memory.html), so code containing calibration experiments can be reexecuted without needing to re-run the experiments. Loading from local SSD disk takes about 1 minute for 6000 experiments.

[^how-to-make-faster]: This could be improved with better optimized implementation of the [hierarchical beta sampling process](code_path-sampling_hierarchical-beta).


##### Effects on caching

Like any [RecordedTask](https://sumatratask.readthedocs.io/en/latest/basics.html), `Calibrate` will record its output to disk. If executed again with exactly the same parameters, instead of evaluating the task again, the result is simply loaded from disk.

In addition, `Calibrate` (or rather `Bemd`, which it calls internally) also uses a faster `joblib.Memory` cache to store intermediate results for each value of $c$ in `c_list`. Because `joblib.Memory` computes its hashes by first pickling its inputs, this cache is neither portable nor suitable for long-term storage: the output of `pickle.dump` may change depending on the machine, OS version, Python version, etc. Therefore this cache should be consider *local* and *short-term*. Nevertheless it is quite useful, because it means that `c_list` can be modified and only the new $c$ values will be computed.

Changing any argument other than `c_list` will invalidate all caches and force all recomputations.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

**Current limitations**
- There is no way to set `ncores` directly; to control the number of cores,
  use the configuration option `config.mp.max_cores`.
  `ncores` will be set to the minimum between `max_cores` and the maximum number allowed by the machine.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Types

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### Input types

To be able to retrieve pasts results, [Tasks](https://sumatratask.readthedocs.io/en/latest/basics.html) rely on their inputs being serializable (i.e. convertible to plain text). Both [*Pydantic*](https://docs.pydantic.dev/latest/) and [*SciTyping*](https://scityping.readthedocs.io) types are supported; *SciTyping* in particular can serialize arbitrary [dataclasses](https://docs.python.org/3/library/dataclasses.html), as long as each of their fields are serializable.

A weaker requirement for an object is to be pickleable. All serializable objects should be pickleable, but many pickleable objects are not serializable. In general, objects need to be pickleable if they are sent to a multiprocessing (MP) subprocess, and serializable if they are written to the disk.

| Requirement | Reason | Applies to |
|-------------|--------|----------|
| Pickleable  | Sent to subprocess | `compute_Bemd` arguments |
| Serializable | Saved to disk | `Calibrate` arguments<br>`CalibrateResult` |
| Hashable    | Used as dict key | items of `models`<br>items of `c_list` |

To satisfy these requirements, the sequence `models` needs to be specified as a frozen dataclass:[^more-formats] Dataclasses for serializability, frozen for hashability. Of course they should also define `__iter__` and `__len__` – see [`EpistemicDist`](code_calibration-distribution) for an example.

[CHECK: I think loss functions are unrestricted now ? Maybe impact on caching ?] The loss functions `QA` and `QB` can be specified as either dataclasses (with a suitable `__call__` method) or [`PureFunction`s](https://scityping.readthedocs.io/en/latest/api/functions.html#scityping.functions.PureFunction). In practice we found dataclasses easier to use.

[^more-formats]: We use dataclasses because they are the easiest to support, but support for other formats could be added in the future.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
SynthPPF = Callable[[np.ndarray[float]], np.ndarray[float]]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The items of the `models` sequence must be functions which take a single argument – the data size $L$ – and return a data set of size $L$:
```{math}
\begin{aligned}
\texttt{data_model}&:& L &\mapsto
      \bigl[(x_1, y_1), (x_2, y_2), \dotsc, (x_L, y_L)\bigr]
\end{aligned} \,.
```
Exactly how the dataset is structured (single array, list of tuples, etc.) is up to the user.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
Dataset = TypeVar("Dataset",
                  bound=Union[np.ndarray,
                              List[np.ndarray],
                              List[Tuple[np.ndarray, np.ndarray]]]
                 )
DataModel = Callable[[int], Dataset]
CandidateModel = Callable[[Dataset], Dataset]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The `riskA` and `riskB` functions take a dataset returned by `data_model` and evaluate the risk $q$ of each sample. They return a vector of length $L$, and their signature depends on the output format of `data_model`:
```{math}
\begin{aligned}
\texttt{risk function}&:& \{(x_i,y_i)\}_{i=1}^L &\mapsto \{q_i\}_{i=1}^L \,.
\end{aligned}
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
RiskFunction = Callable[[Dataset], np.ndarray]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### Result type

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Calibration results are returned as a [record array](https://numpy.org/doc/stable/user/basics.rec.html#record-arrays) with fields `Bemd` and `Bepis`. Each row in the array corresponds to one data model, and there is one array per $c$ value. So a `CalibrateResult` object is a dictionary which looks something like the following:

+++ {"editable": true, "slideshow": {"slide_type": ""}}

```{math}
\begin{alignedat}{4}  % Would be nicer with nested {array}, but KaTeX doesn’t support vertical alignment
&\texttt{CalibrateResult}:\qquad & \{ c_1: &\qquad&  \texttt{Bemd} &\quad& \texttt{Bepis} \\
  &&&&  0.24    && 0 \\
  &&&&  0.35    && 1 \\
  &&&&  0.37    && 0 \\
  &&&&  0.51    && 1 \\
  && c_2: &\qquad&  \texttt{Bemd} &\quad& \texttt{Bepis} \\
  &&&&  0.11    && 0 \\
  &&&&  0.14    && 0 \\
  &&&&  0.22    && 0 \\
  &&&&  0.30    && 1 \\
  &&\vdots \\
  &&\}
\end{alignedat}
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
calib_point_dtype = np.dtype([("Bemd", float), ("Bepis", bool)])
CalibrateResult = dict[float, np.ndarray[calib_point_dtype]]
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
class CalibrateOutput(TaskOutput):
    """Compact format used to store task results to disk.
    Use `task.unpack_result` to convert to a `CalibrateResult` object.
    """
    Bemd : List[float]
    Bepis: List[float]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Functions for the calibration experiment

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Below we define the two functions to compute $\Bemd{}$ and $\Bepis{}$; these will be the abscissa and ordinate in the calibration plot.
Both functions take an arguments a data generation model, risk functions for candidate models $A$ and $B$, and a number of data points to generate.

- $\Bemd{}$ needs to be recomputed for each value of $c$, so we also pass $c$ as a parameter. $\Bemd{}$ computations are relatively expensive, and there are a lot of them to do during calibration, so we want to dispatch `compute_Bemd` to different multiprocessing (MP) processes. This has two consequences:

  - The `multiprocessing.Pool.imap` function we use to dispatch function calls can only iterate over one argument. To accomodate this, we combine the data model and $c$ value into a tuple `datamodel_c`, which is unpacked within the `compute_Bemd` function.
  - All arguments should be pickleable, as pickle is used to send data to subprocesses.

- $\Bepis{}$ only needs to be computed once per data model. $\Bepis{}$ is also typically cheap (unless the data generation model is very complicated), so it is not worth dispatching to an MP subprocess.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
def compute_Bemd(i_ω_c: Tuple[int, Experiment, float],
                 #datamodel_risk_c: Tuple[int,DataModel,CandidateModel,CandidateModel,RiskFunction,RiskFunction,float],
                 #riskA: RiskFunction, riskB: RiskFunction,
                 #synth_ppfA: SynthPPF, synth_ppfB: SynthPPF,
                 #candidate_model_A: CandidateModel, candidate_model_B: CandidateModel,
                 Ldata):
    """
    Wrapper for `emdcmp.Bemd`:
    - Unpack `datamodel_c` into `data_mode
    - Instantiates models using parameters in `Θtup_c`.
    - Constructs log-probability functions for `MtheoA` and `MtheoB`.
    - Generates synthetic observed data using `Mtrue`.
    - Calls `emdcmp.Bemd`

    """
    ## Unpack arg 1 ##  (pool.imap requires iterating over one argument only)
    i, ω, c = i_ω_c

    ## Generate observed data ##
    logger.debug(f"Compute Bemd - Generating {Ldata} data points."); t1 = time.perf_counter()
    data = ω.data_model(Ldata)                                     ; t2 = time.perf_counter()
    logger.debug(f"Compute Bemd - Done generating {Ldata} data points. Took {t2-t1:.2f} s")

    ## Construct synthetic quantile functions ##
    synth_ppfA = emd.make_empirical_risk_ppf(ω.QA(ω.candidateA(data)))
    synth_ppfB = emd.make_empirical_risk_ppf(ω.QB(ω.candidateB(data)))

    ## Construct mixed quantile functions ##
    mixed_ppfA = emd.make_empirical_risk_ppf(ω.QA(data))
    mixed_ppfB = emd.make_empirical_risk_ppf(ω.QB(data))

    ## Draw sets of expected risk values (R) for each model ##
                     
    # Silence sampling warnings: Calibration involves evaluating Bemd for models far from the data distribution, which require more
    # than 1000 path samples to evaluate the path integral within the default margin.
    # The further paths are from the most likely one, the more likely they are to trigger numerical warnings.
    # This is expected, so we turn off warnings to avoid spamming the console.

    logger.debug("Compute Bemd - Generating R samples"); t1 = time.perf_counter()
    
    emdlogger = logging.getLogger("emdcmp.emd")
    emdlogginglevel = emdlogger.level
    emdlogger.setLevel(logging.ERROR)

    # NB: Calibration explores some less well-fitted regions, so keeping `res` and `M` high is worthwhile
    #     (Otherwise we get poor Bemd estimates and need more data.)
    RA_lst = emd.draw_R_samples(mixed_ppfA, synth_ppfA, c=c)
    RB_lst = emd.draw_R_samples(mixed_ppfB, synth_ppfB, c=c)

    # Reset logging level as it was before
    emdlogger.setLevel(emdlogginglevel)

    t2 = time.perf_counter()
    logger.debug(f"Compute Bemd - Done generating R samples. Took {t2-t1:.2f} s")
                     
    ## Compute the EMD criterion ##
    Bemd = np.less.outer(RA_lst, RB_lst).mean()

    ## Return alongside the experiment key
    return (i, ω, c), Bemd
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
def compute_Bepis(data_model, QA, QB, Linf):
    """Compute the true Bepis (using a quasi infinite number of samples)"""
    
    # Generate samples
    logger.debug(f"Compute Bepis – Generating 'infinite' dataset with {Linf} data points"); t1 = time.perf_counter()
    data = data_model(Linf)
    t2 = time.perf_counter()
    logger.debug(f"Compute Bepis – Done generating 'infinite' dataset. Took {t2-t1:.2f} s")
    
    # Compute Bepis
    logger.debug("Compute Bepis – Evaluating expected risk on 'infinite' dataset"); t1 = time.perf_counter()
    RA = QA(data).mean()
    RB = QB(data).mean()
    t2 = time.perf_counter()
    logger.debug(f"Compute Bepis – Done evaluating risk. Took {t2-t1:.2f} s")
    return RA < RB
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
def compute_Bemd_and_maybe_Bepis(i_ω_c, Ldata, Linf, c_conf):
    """Wrapper which calls both `compute_Bemd` and `compute_Bepis`.
    The latter is only called if `c` matches `c_conf`. 
    
    The reason for this wrapper is to better utilize multiprocessing threads,
    by executing Bepis with the same MP threads as Bemd while still ensuring
    that Bepis is not executed more often than needed.
    Since Bemd is executed once for every `c` value in `c_list`, we choose
    one value in `c_list` and make that special: whenever we compute Bemd for
    that `c`, we also compute Bepis.

    This has three related benefits:
    - If there is caching that might reuse computations between compute_Bemd
      and compute_Bepis (e.g. a data generation function decorated with @cache),
      this increases the chances 
    - It avoids having to execute Bepis in the main process, which can have
      adverse effects if Bepis involves significant computation. If we have
      n cores and n MP processes, then having computation occuring in the main
      process is like having n+1 MP processes, which will lead to inefficient
      switching.
    - If the Bepis computations are so significant (less than n times faster
      than Bemd), then they become a bottleneck and cause `imap` to accumulate
      a queue of results. Since this is effectively an unbounded cache, if
      those results require substantial memory, this can cause the entire
      computation to crash because it exceeds available memory.
    """
    # NB: Since `data_model` is consumed within this function, even if there
    #     were a bottleneck with imap, it should not exceed memory:
    #     i, c, Bemd and Bepis are all small scalars.
    (i, ω, c), Bemd = compute_Bemd(i_ω_c, Ldata)
    Bepis = compute_Bepis(ω.data_model, ω.QA, ω.QB, Linf) if c == c_conf \
            else None
    return i, c, Bemd, Bepis
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Task definition

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
@RecordedTask
class Calibrate:

    def __call__(
        self,
        c_list     : List[float],
        experiments: Dataclass,   # Iterable of Experiment elements
        Ldata      : int,
        Linf       : int,
        ) -> CalibrateOutput:
        """
        Run a calibration experiment using the models listed in `models`.
        Data models must be functions taking a single argument – an integer – and
        returning a dataset with that many samples. They should be “ready to use”;
        in particular, their random number generator should already be properly seeded
        to avoid correlations between different models in the list.
        
        Parameters
        ----------
        c_list:
        
        experiments: Dataclass following the pattern of `EpistemicDist`.
            Therefore also an iterable of data models to use for calibration;
            when iterating, each element should be compatible with `Experiment` type.
            See `EpistemicDist` for more details.
            Each experiment will result in one (Bepis, Bemd) pair in the output results.
            If this iterable is sized, progress bars will estimate the remaining compute time.
        
        Ldata: Number of data points from the true model to generate when computing Bemd.
            This should be chosen commensurate with the size of the dataset that will be analyzed,
            in order to accurately mimic data variability.
        Linf: Number of data points from the true model to generate when computing `Bepis`.
            This is to emulate an infinitely large data set, and so should be large
            enough that numerical variability is completely suppressed.
            Choosing a too small value for `Linf` will add noise to the Bepis estimate,
            which would need to compensated by more calibration experiments.
            Since generating more samples is generally cheaper than performing more
            experiments, it is also generally preferable to choose rather large `Linf`
            values.

        .. Important:: An appropriate value of `Linf` will depend on the models and
           how difficult they are to differentiate; it needs to be determined empirically.
        """
        pass
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### Prepare the runs

Bind arguments to the `Bemd` function, so it only takes one argument (`i_ω_c`) as required by `imap`.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
        # compute_Bemd_partial = partial(compute_Bemd, Ldata=Ldata)
        compute_partial = partial(compute_Bemd_and_maybe_Bepis,
                                  Ldata=Ldata, Linf=Linf, c_conf=c_list[0])
```

Define dictionaries into which we will accumulate the results of the $B^{\mathrm{EMD}}$ and $B_{\mathrm{conf}}$ calculations.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
        Bemd_results = {}
        Bepis_results = {}
```

- Set the iterator over parameter combinations (we need two identical ones)
- Set up progress bar.
- Determine the number of multiprocessing cores we will use.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
        try:
            N = len(experiments)
        except (TypeError, AttributeError):  # Typically TypeError, but AttributeError seems also plausible
            logger.info("Data model iterable has no length: it will not be possible to estimate the remaining computation time.")
            total = None
        else:
            total = N*len(c_list)
        progbar = tqdm(desc="Calib. experiments", total=total)
        ncores = psutil.cpu_count(logical=False)
        ncores = min(ncores, total, config.mp.max_cores)
```

#### Run the experiments
Since there are a lot of them, and they each take a few minutes, we use multiprocessing to run them in parallel.

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
        ω_c_gen = ((i, ω, c)  # i is used as an id for each different model/Qs set
                   for i, ω in enumerate(experiments)
                   for c in c_list)

        if ncores > 1:
            with mp.Pool(ncores, maxtasksperchild=config.mp.maxtasksperchild) as pool:
                # Chunk size calculated following mp.Pool's algorithm (See https://stackoverflow.com/questions/53751050/multiprocessing-understanding-logic-behind-chunksize/54813527#54813527)
                # (Naive approach would be total/ncores. This is most efficient if all taskels take the same time. Smaller chunks == more flexible job allocation, but more overhead)
                chunksize, extra = divmod(N, ncores*6)
                if extra:
                    chunksize += 1
                Bemd_Bepis_it = pool.imap_unordered(compute_partial, ω_c_gen,
                                                    chunksize=chunksize)
                for (i, c, Bemd, Bepis) in Bemd_Bepis_it:
                    progbar.update(1)        # Updating first more reliable w/ ssh
                    Bemd_results[i, c] = Bemd
                    if Bepis is not None:
                        Bepis_results[i] = Bepis
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Variant without multiprocessing:

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
        else:
            Bemd_Bepis_it = (compute_partial(arg) for arg in ω_c_gen)
            for (i, c, Bemd, Bepis) in Bemd_Bepis_it:
                progbar.update(1)
                Bemd_results[i, c] = Bemd
                if Bepis is not None:
                    Bepis_results[i] = Bepis
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### Cleanup

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
        progbar.close()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

#### Package the results

If we serialize the whole dict, most of the space is taken up by serializing the models in the keys. Not only is this wasteful – we can easily recreate them from the task arguments – but it also makes deserializing the results quite slow.
So instead we return just the values as a list, and provide an `unpack_result` method which reconstructs the result dictionary.
:::{caution}
This assumes that we get the same models when we rebuild them within `unpack_result`.
Two ways this assumption can be violated:
- The user’s code for `experiments` has changed. (Unless the user used a type
  which serialises to completely self-contained data.)
- The model generation is non-reproducible. (E.g. it uses an unseeded RNG).
:::

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution, remove-cell]
---
        # NB: Don’t just use list(Bemd_results.values()): order of dictionary is not guaranteed
        return dict(Bemd =[Bemd_results [i,c] for i in range(len(experiments)) for c in c_list],
                    Bepis=[Bepis_results[i]   for i in range(len(experiments))])
```

+++ {"tags": ["remove-cell"], "editable": true, "slideshow": {"slide_type": ""}}

> **END OF `Calibrate.__call__`**

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [skip-execution]
---
    def unpack_results(self, result: Calibrate.Outputs.result_type
                      ) -> CalibrateResult:
        """
        Take the compressed result exported by the task, and recreate the
        dictionary structure of a `CalibrateResult`, where experiments are
        organized by their value of `c`.
        """
        assert len(result.Bemd) == len(self.c_list) * len(result.Bepis), \
            "`result` argument seems not to have been created with this task."

        # Reconstruct the dictionary as it was at the end of task execution
        Bemd_dict = {}; Bemd_it = iter(result.Bemd)
        Bepis_dict = {}; Bepis_it = iter(result.Bepis)
        for i in range(len(result.Bepis)):         # We don’t actually need the models
            Bepis_dict[i] = next(Bepis_it)         # => we just use integer ids
            for c in self.taskinputs.c_list:       # This avoids unnecessarily
                Bemd_dict[i, c] = next(Bemd_it)  # instantiating models.

        # Package results into record arrays – easier to sort and plot
        calib_curve_data = {c: [] for c in self.taskinputs.c_list}
        for i, c in Bemd_dict:
            calib_curve_data[c].append(
                (Bemd_dict[i, c], Bepis_dict[i]) )

        return {c: np.array(calib_curve_data[c], dtype=calib_point_dtype)
                for c in self.taskinputs.c_list}
```
