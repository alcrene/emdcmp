# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python (emd-paper)
#     language: python
#     name: emd-paper
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ---
# math:
#   '\Bconf' : 'B^{\mathrm{conf}}_{#1}'
#   '\Bemd'  : 'B_{#1}^{\mathrm{EMD}}'
# ---

# %% editable=true slideshow={"slide_type": ""} tags=["hide-input"]
from __future__ import annotations

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Tasks
#
# Running experiments via [SumatraTasks](https://sumatratask.readthedocs.io/en/latest/basics.html) has two purposes:
# - Maintaining an electronic lab book: recording all input/code/output triplets, along with a bunch of metadata to ensure reproducibility (execution date, code versions, etc.)
# - Avoid re-running calculations, with hashes that are both portable and long-term stable.

# %% tags=["active-ipynb"] editable=true slideshow={"slide_type": ""}
# from config import config   # Notebook

# %% tags=["active-py"] editable=true slideshow={"slide_type": ""} raw_mimetype=""
from .config import config  # Python script

# %% editable=true slideshow={"slide_type": ""}
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

import emd_falsify as emd

# %% editable=true slideshow={"slide_type": ""}
logger = logging.getLogger(__name__)

# %% editable=true slideshow={"slide_type": ""}
__all__ = ["Calibrate", "CalibrationDist", "CalibrateOutput"]


# %% [markdown] editable=true slideshow={"slide_type": ""}
# (code_calibration-distribution)=
# ## Calibration distribution

# %% editable=true slideshow={"slide_type": ""}
@dataclass(frozen=True)
class CalibrationDist(abc.ABC):
    """Generic template for a calibration distribution:
    
    in effect a calibration distribution with no calibration parameters.
    For actual use you need to subclass `CalibrationDist` and extend it with 
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
    def __iter__(self):
        raise NotImplementedError
        # rng = <create & seed an RNG using the dist parameters as entropy>
        # for n in range(self.N):
        #     <draw calibration params using rng>
        #     yield <data model>

    def __len__(self):
        return self.N
    
    def generate(self, N: int):
        """Return a copy of CalibrationDist which will yield `N` models.
        
        :param:N: Number of models to return.
        """
        return replace(self, N=N)


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Calibration task

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Task parameters
#
# | Parameter | Description |
# |-----------|-------------|
# | `c_list` | The values of $c$ we want to test. |
# | `data_models` | Sequence of $N$ data generation models drawn from a calibration distribution. Typically, but not necessarily, a subclass of `CalibrationDist`: any dataclass satisfying the requirements listed in `CalibrationDist` is accepted. |
# | `riskA` | Risk function for candidate model $A$. |
# | `riskA` | Risk function for candidate model $B$. |
# | `synth_risk_ppfA` | Synthetic PPF of the risk of candidate model $A$. Almost always an instance of `scipy.interpolate.interp1d`. |
# | `synth_risk_ppfB` | Synthetic PPF of the risk of candidate model $B$. Almost always an instance of `scipy.interpolate.interp1d`. |
# | `Ldata` | Data set size used to construct the empirical PPF for models $A$ and $B$. Ideally commensurate with the actual data set used to assess models. |
# | `Linf` | Data set size considered equivalent to "infinite". Used to compute $\B^{\mathrm{conf}}$ |
#
# The value of $N$ is determined from `len(data_models)`, so the `data_models` iterable should define its length.
#
# #### Config values:
#
# | Parameter | Description |
# |-----------|-------------|
# | `ncores`  |  Number of CPU cores to use. |
#
# ##### Effects on compute time
#
# The total number of experiments will be
# $$N \times \lvert\mathtt{c\_list}\rvert \times \text{(\# parameter set distributions)} \,.$$
# In the best scenario, one can expect compute times to be 2.5 minutes / experiment. So expect this to take a few hours.
#
# Results are cached on-disk with [joblib.Memory](https://joblib.readthedocs.io/en/latest/memory.html), so this notebook can be reexecuted without re-running the experiments. Loading from disk takes about 1 minute for 6000 experiments.
#
# ##### Effects on caching
#
# Like any [RecordedTask](https://sumatratask.readthedocs.io/en/latest/basics.html), `Calibrate` will record its output to disk. If executed again with exactly the same parameters, instead of evaluating the task again, the result is simply loaded from disk.
#
# In addition, `Calibrate` (or rather `Bemd`, which it calls internally) also uses a faster `joblib.Memory` cache to store intermediate results for each value of $c$ in `c_list`. Because `joblib.Memory` computes its hashes by first pickling its inputs, this cache is neither portable nor suitable for long-term storage: the output of `pickle.dump` may change depending on the machine, OS version, Python version, etc. Therefore this cache should be consider *local* and *short-term*. Nevertheless it is quite useful, because it means that `c_list` can be modified and only the new $c$ values will be computed.
#
# Changing any argument other than `c_list` will invalidate all caches and force all recomputations.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# **Current limitations**
# - `ncores` depends on `config.mp.max_cores`, but is determined automatically.
#   No way to control via parameter.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Types

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Input types
#
# To be able to retrieve pasts results, [Tasks](https://sumatratask.readthedocs.io/en/latest/basics.html) rely on their inputs being serializable (i.e. convertible to plain text). Both [*Pydantic*](https://docs.pydantic.dev/latest/) and [*SciTyping*](https://scityping.readthedocs.io) types are supported; *SciTyping* in particular can serialize arbitrary [dataclasses](https://docs.python.org/3/library/dataclasses.html), as long as each of their fields are serializable.
#
# A weaker requirement for an object is to be pickleable. All serializable objects should be pickleable, but many pickleable objects are not serializable. In general, objects need to be pickleable if they are sent to a multiprocessing (MP) subprocess, and serializable if they are written to the disk.
#
# | Requirement | Reason | Applies to |
# |-------------|--------|----------|
# | Pickleable  | Sent to subprocess | `compute_Bemd` arguments |
# | Serializable | Saved to disk | `Calibrate` arguments<br>`CalibrateResult` |
# | Hashable    | Used as dict key | items of `data_models`<br>items of `c_list` |
#
# To satisfy these requirements, the sequence `data_models` needs to be specified as a frozen dataclass:[^more-formats] Dataclasses for serializability, frozen for hashability. Of course they should also define `__iter__` and `__len__` – see [`CalibrationDist`](code_calibration-distribution) for an example.
#
# The `riskA` and `riskB` functions can be specified as either dataclasses (with a suitable `__call__` method) or [`PureFunction`s](https://scityping.readthedocs.io/en/latest/api/functions.html#scityping.functions.PureFunction). In practice we found dataclasses easier to use.
#
# For `synth_risk_ppfA`, the `scipy.interpolate.interp1d` is to our knowledge always the most appropriate. This is the type returned by `emd.make_empirical_ppf` and we have special support to serialize this type.
#
# [^more-formats]: We use dataclasses because they are the easiest to support, but support for other formats could be added in the future.

# %% editable=true slideshow={"slide_type": ""}
SynthPPF = Callable[[np.ndarray[float]], np.ndarray[float]]

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The items of the `data_models` sequence must be functions which take a single argument – the data size $L$ – and return a data set of size $L$:
# $$\begin{aligned}
# \texttt{data\_model}&:& L &\mapsto
#       \bigl[(x_1, y_1), (x_2, y_2), \dotsc, (x_L, y_L)\bigr]
# \end{aligned} \,. $$
# Exactly how the dataset is structured (single array, list of tuples, etc.) is up to the user.

# %% editable=true slideshow={"slide_type": ""}
Dataset = TypeVar("Dataset",
                  bound=Union[np.ndarray,
                              List[np.ndarray],
                              List[Tuple[np.ndarray, np.ndarray]]]
                 )
DataModel = Callable[[int], Dataset]

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The `riskA` and `riskB` functions take a dataset returned by `data_model` and evaluate the risk $q$ of each sample. They return a vector of length $L$, and their signature depends on the output format of `data_model`:
# $$\begin{aligned}
# \texttt{risk function}&:& \{(x_i,y_i)\}_{i=1}^L &\mapsto \{q_i\}_{i=1}^L \,.
# \end{aligned}$$

# %% editable=true slideshow={"slide_type": ""}
RiskFunction = Callable[[Dataset], np.ndarray]

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Result type

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Calibration results are returned as a [record array](https://numpy.org/doc/stable/user/basics.rec.html#record-arrays) with fields `Bemd` and `Bconf`. Each row in the array corresponds to one data model, and there is one array per $c$ value. So a `CalibrateResult` object is a dictionary which looks something like the following:

# %% [markdown] editable=true slideshow={"slide_type": ""}
# $$
# \begin{alignedat}{4}  % Would be nicer with nested {array}, but KaTeX doesn’t support vertical alignment
# &\texttt{CalibrateResult}:\qquad & \{ c_1: &\qquad&  \texttt{Bemd} &\quad& \texttt{Bconf} \\
#   &&&&  0.24    && 0 \\
#   &&&&  0.35    && 1 \\
#   &&&&  0.37    && 0 \\
#   &&&&  0.51    && 1 \\
#   && c_2: &\qquad&  \texttt{Bemd} &\quad& \texttt{Bconf} \\
#   &&&&  0.11    && 0 \\
#   &&&&  0.14    && 0 \\
#   &&&&  0.22    && 0 \\
#   &&&&  0.30    && 1 \\
#   &&\vdots \\
#   &&\}
# \end{alignedat}$$

# %% editable=true slideshow={"slide_type": ""}
calib_point_dtype = np.dtype([("Bemd", float), ("Bconf", bool)])
CalibrateResult = dict[float, np.ndarray[calib_point_dtype]]


# %% editable=true slideshow={"slide_type": ""}
class CalibrateOutput(TaskOutput):
    """Compact format used to store task results to disk.
    Use `task.unpack_result` to convert to a `CalibrateResult` object.
    """
    Bemd : List[float]
    Bconf: List[float]


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Functions for the calibration experiment

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Below we define the two functions to compute $\Bemd{}$ and $\Bconf{}$; these will be the abscissa and ordinate in the calibration plot.
# Both functions take an arguments a data generation model, risk functions for candidate models $A$ and $B$, and a number of data points to generate.
#
# - $\Bemd{}$ needs to be recomputed for each value of $c$, so we also pass $c$ as a parameter. $\Bemd{}$ computations are relatively expensive, and there are a lot of them to do during calibration, so we want to dispatch `compute_Bemd` to different multiprocessing (MP) processes. This has two consequences:
#
#   - The `multiprocessing.Pool.imap` function we use to dispatch function calls can only iterate over one argument. To accomodate this, we combine the data model and $c$ value into a tuple `datamodel_c`, which is unpacked within the `compute_Bemd` function.
#   - All arguments should be pickleable, as pickle is used to send data to subprocesses.
#
# - $\Bconf{}$ only needs to be computed once per data model. $\Bconf{}$ is also typically cheap (unless the data generation model is very complicated), so it is not worth dispatching to an MP subprocess.  

# %% editable=true slideshow={"slide_type": ""}
def compute_Bemd(datamodel_c: Tuple[DataModel,float],
                 riskA: RiskFunction, riskB: RiskFunction,
                 synth_ppfA: SynthPPF, synth_ppfB: SynthPPF,
                 Ldata):
    """
    Wrapper for `emd_falsify.Bemd`:
    - Unpack `datamodel_c` into `data_mode
    - Instantiates models using parameters in `Θtup_c`.
    - Constructs log-probability functions for `MtheoA` and `MtheoB`.
    - Generates synthetic observed data using `Mtrue`.
    - Calls `emd_falsify.Bemd`

    `Mptrue`, `Metrue`, `Mptheo` and `Metheo` should be generic models:
    they are functions accepting parameters, and returning frozen models
    with fixed parameters.
    """
    ## Unpack arg 1 ##  (pool.imap requires iterating over one argument only)
    data_model, c = datamodel_c

    ## Generate observed data ##
    logger.debug(f"Compute Bemd - Generating {Ldata} data points."); t1 = time.perf_counter()
    data = data_model(Ldata)                                      ; t2 = time.perf_counter()
    logger.debug(f"Compute Bemd - Done generating {Ldata} data points. Took {t2-t1:.2f} s")

    ## Construct mixed quantile functions ##
    mixed_ppfA = emd.make_empirical_risk_ppf(riskA(data))
    mixed_ppfB = emd.make_empirical_risk_ppf(riskB(data))

    ## Draw sets of expected risk values (R) for each model ##
                     
    # Silence sampling warnings: Calibration involves evaluating Bemd for models far from the data distribution, which require more
    # than 1000 path samples to evaluate the path integral within the default margin.
    # The further paths are from the most likely one, the more likely they are to trigger numerical warnings.
    # This is expected, so we turn off warnings to avoid spamming the console.

    logger.debug("Compute Bemd - Generating R samples"); t1 = time.perf_counter()
    
    emdlogger = logging.getLogger("emd_falsify.emd")
    emdlogginglevel = emdlogger.level
    emdlogger.setLevel(logging.ERROR)

    RA_lst = emd.draw_R_samples(mixed_ppfA, synth_ppfA, c=c)
    RB_lst = emd.draw_R_samples(mixed_ppfB, synth_ppfB, c=c)

    # Reset logging level as it was before
    emdlogger.setLevel(emdlogginglevel)

    t2 = time.perf_counter()
    logger.debug(f"Compute Bemd - Done generating R samples. Took {t2-t1:.2f} s")
                     
    ## Compute the EMD criterion ##
    return np.less.outer(RA_lst, RB_lst).mean()

# %% editable=true slideshow={"slide_type": ""}
def compute_Bconf(data_model, riskA, riskB, Linf):
    """Compute the true Bconf (using a quasi infinite number of samples)"""
    
    # Generate samples
    logger.debug(f"Compute Bconf – Generating 'infinite' dataset with {Linf} data points"); t1 = time.perf_counter()
    data = data_model(Linf)
    t2 = time.perf_counter()
    logger.debug(f"Compute Bconf – Done generating 'infinite' dataset. Took {t2-t1:.2f} s")
    
    # Compute Bconf
    logger.debug("Compute Bconf – Evaluating expected risk on 'infinite' dataset"); t1 = time.perf_counter()
    RA = riskA(data).mean()
    RB = riskB(data).mean()
    t2 = time.perf_counter()
    logger.debug(f"Compute Bconf – Done evaluating risk. Took {t2-t1:.2f} s")
    return RA < RB


# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Task definition

# %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
@RecordedTask
class Calibrate:

    def __call__(
        self,
        c_list     : List[float],
        #data_models: Sequence[DataModel],
        data_models: Dataclass,
        #riskA     : RiskFunction
        riskA      : Union[Dataclass,PureFunction],
        riskB      : Union[Dataclass,PureFunction],
        #synth_risk_ppf: SynthPPF
        synth_risk_ppfA  : Union[emd.interp1d, PureFunction],
        synth_risk_ppfB  : Union[emd.interp1d, PureFunction],
        Ldata      : int,
        Linf       : int,
        ) -> CalibrateOutput:
        """
        Run a calibration experiment using the models listed in `data_models`.
        Data models must be functions taking a single argument – an integer – and
        returning a dataset with that many samples. They should be “ready to use”;
        in particular, their random number generator should already be properly seeded
        to avoid correlations between different models in the list.
        
        Parameters
        ----------
        c_list:
        
        data_models: Dataclass following the pattern of `CalibrationDist`.
            Therefore also an iterable of data models to use for calibration.
            See `CalibrationDist` for more details.
            Each data model will result in one (Bconf, Bemd) pair in the output results.
            If this iterable is sized, progress bars will estimate the remaining compute time.
        
        riskA, riskB:
        
        synth_riskA, synth_riskB:
        
        Ldata: Number of data points from the true model to generate when computing Bemd.
            This should be chosen commensurate with the size of the dataset that will be analyzed,
            in order to accurately mimic data variability.
        Linf: Number of data points from the true model to generate when computing `Bconf`.
            This is to emulate an infinitely large data set, and so should be large
            enough that numerical variability is completely suppressed.
            Choosing a too small value for `Linf` will add noise to the Bconf estimate,
            which would need to compensated by more calibration experiments.
            Since generating more samples is generally cheaper than performing more
            experiments, it is also generally preferable to choose rather large `Linf`
            values.

        .. Important:: An appropriate value of `Linf` will depend on the models and
           how difficult they are to differentiate; it needs to be determined empirically.
        """
        pass


# %% [markdown] editable=true slideshow={"slide_type": ""}
# Bind arguments to the `Bemd` function, so it only take one argument (`datamodel_c`) as required by `imap`.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        compute_Bemd_partial = partial(
            compute_Bemd, riskA=riskA, riskB=riskB,
                          synth_ppfA=synth_risk_ppfA, synth_ppfB=synth_risk_ppfB,
                          Ldata=Ldata)

# %% [markdown]
# Define dictionaries into which we will accumulate the results of the $B^{\mathrm{EMD}}$ and $B_{\mathrm{conf}}$ calculations.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        Bemd_results = {}
        Bconf_results = {}

# %% [markdown]
# - Set the iterator over parameter combinations (we need two identical ones)
# - Set up progress bar.
# - Determine the number of multiprocessing cores we will use.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        try:
            N = len(data_models)
        except (TypeError, AttributeError):  # Typically TypeError, but AttributeError seems also plausible
            logger.info("Data model iterable has no length: it will not be possible to estimate the remaining computation time.")
            total = None
        else:
            total = N*len(c_list)
        progbar = tqdm(desc="Calib. experiments", total=total)
        ncores = psutil.cpu_count(logical=False)
        ncores = min(ncores, total, config.mp.max_cores)

# %% [markdown]
# Run the experiments. Since there are a lot of them, and they each take a few minutes, we use multiprocessing to run them in parallel.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        if ncores > 1:
            with mp.Pool(ncores) as pool:
                # Chunk size calculated following Pool's algorithm (See https://stackoverflow.com/questions/53751050/multiprocessing-understanding-logic-behind-chunksize/54813527#54813527)
                # (Naive approach would be total/ncores. This is most efficient if all taskels take the same time. Smaller chunks == more flexible job allocation, but more overhead)
                chunksize, extra = divmod(N, ncores*6)
                if extra:
                    chunksize += 1
                Bemd_it = pool.imap(compute_Bemd_partial,
                                    self.model_c_gen(Bemd_results, Bconf_results),
                                    chunksize=chunksize)
                for (data_model, c), Bemd_res in zip(                                     # NB: Both `models_c_gen` generators
                        self.model_c_gen(Bemd_results, Bconf_results),               # always yield the same tuples,
                        Bemd_it):                                                     # because we only update Bemd_results
                    progbar.update(1)        # Updating first more reliable w/ ssh    # after drawing from the second generator
                    Bemd_results[data_model, c] = Bemd_res
                    if data_model not in Bconf_results:
                        Bconf_results[data_model] = compute_Bconf(data_model, riskA, riskB, Linf)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Variant without multiprocessing:

        # %% editable=true slideshow={"slide_type": ""}
        else:
            Bemd_it = (compute_Bemd_partial(arg)
                       for arg in self.model_c_gen(Bemd_results, Bconf_results))
            for (data_model, c), Bemd_res in zip(                                     # NB: Both `model_c_gen` generators
                    self.model_c_gen(Bemd_results, Bconf_results),               # always yield the same tuples,
                    Bemd_it):                                                     # because we only update Bemd_results
                progbar.update(1)        # Updating first more reliable w/ ssh    # after drawing from the second generator
                Bemd_results[data_model, c] = Bemd_res
                if data_model not in Bconf_results:
                    Bconf_results[data_model] = compute_Bconf(data_model, riskA, riskB, Linf)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Close progress bar:

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        progbar.close()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### Result format
#
# If we serialize the whole dict, most of the space is taken up by serializing the data_models in the keys. Not only is this wasteful – we can easily recreate them with `model_c_gen` – but it also makes deserializing the results quite slow.
# So instead we return just the values as a list, and provide an `unpack_result` method which reconstructs the result dictionary.

        # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
        return dict(Bemd =list(Bemd_results.values()),
                    Bconf=list(Bconf_results.values()))

# %% [markdown] editable=true slideshow={"slide_type": ""}
# > **END OF `Calibrate.__call__`**

    # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
    def unpack_results(self, result: Calibrate.Outputs.result_type
                      ) -> CalibrateResult:
        # Reconstruct the dictionary as it was at the end of task execution
        Bemd_dict = {}; Bemd_it = iter(result.Bemd)
        Bconf_dict = {}; Bconf_it = iter(result.Bconf)
        for data_model in self.taskinputs.data_models:
            Bconf_dict[data_model] = next(Bconf_it)
            for c in self.taskinputs.c_list:
                Bemd_dict[(data_model, c)] = next(Bemd_it)
        # Package results into a record arrays – much easier to sort and plot
        calib_curve_data = {c: [] for c in self.taskinputs.c_list}
        for data_model, c in Bemd_dict:
            calib_curve_data[c].append(
                (Bemd_dict[(data_model, c)], Bconf_dict[data_model]) )

        #return UnpackedCalibrateResult(Bemd=Bemd_dict, Bconf=Bconf_dict)
        return {c: np.array(calib_curve_data[c], dtype=calib_point_dtype)
                for c in self.taskinputs.c_list}

# %% [markdown] editable=true slideshow={"slide_type": ""}
# #### (Model, $c$) generator
#
# `task.model_c_gen` yields combined `(data_model, c)` tuples by iterating over both `data_models` and `c_list`. The reason we implement this in its own method is so we can recreate the sequence of models and $c$ values in `unpack_results`: this way we only need to store the sequence of $\Bemd{}$ and $\Bconf{}$ values for each (model, $c$) pair, but not the models themselves.

    # %% editable=true slideshow={"slide_type": ""} tags=["skip-execution"]
    def model_c_gen(self, Bemd_results: "dict|set", Bconf_results: "dict|set"):
        """Return an iterator over data models and c values.
        The two additional arguments `Bemd_results` and `Bconf_results` should be sets of
        *already computed* results. At the moment this is mostly a leftover from a previous
        implementation, before this function was made a *Task* — in the current 
        implementation, the task always behaves as though empty dictionaries were passed.
        """
        for data_model in self.taskinputs.data_models:    # Using taskinputs allows us to call `model_c_gen`
            for c in self.taskinputs.c_list:             # after running the task to recreate the keys.
                if (data_model, c) not in Bemd_results:
                    yield (data_model, c)
                else:                                    # Skip results which are already loaded
                    assert data_model in Bconf_results
