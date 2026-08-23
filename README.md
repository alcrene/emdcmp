# EMD model comparison library

The EMD (empirical model discrepancy) is a method for comparing models based both on their expected risk, and the uncertainty of that risk across experiments.
It is described in [this publication](https://alcrene.github.io/emd-paper); it’s main features are:
- **Symmetric**: All models are treated the same. (There is no preferred null model.)
  A corollary is that the test works for any number of models.
- **Specific**: Models are compared for particular parameter sets. In particular, the different models may all be the same equations but with different parameters.
- **Dimension agnostic**: Models are compared based on their quantile function, which is always 1d. So the method scales well to high-dimensional problems.
- **Designed for real-world data**: A “true” model is not required: all models are assumed to be imperfect.
- **Designed for reproducibility**: Comparisons are based on a model’s ability to generalize.
- **Designed for big data**: Comparisons assume a standard machine learning pipeline with separate training and test datasets.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13287993.svg)](https://doi.org/10.5281/zenodo.13287993)

## Problem requirements

The main requirement to be able to compute the EMD criterion are:

- Observation data.
- A loss function.
- At least two models to compare.
- The ability to use the models to generate synthetic samples.

The models can take any form; they can even be blackbox deep neural networks.

## Installation

    pip install emdcmp

## Usage

The short form for comparing two models is

```python
from emdcmp import Bemd, make_empirical_risk, draw_R_samples

# Load data into `data`
# Define `modelA`, `modelB`
# Define `lossA`, `lossB` : functions
# Define `Lsynth` : int
# Define `c` : float

synth_ppfA = make_empirical_risk(lossA(modelA.generate(Lsynth)))
synth_ppfB = make_empirical_risk(lossB(modelB.generate(Lsynth)))
mixed_ppfA = make_empirical_risk(lossA(data))
mixed_ppfB = make_empirical_risk(lossB(data))

Bemd(mixed_ppfA, mixed_ppfB, synth_ppfA, synth_ppfB, c=c)
```

The function `Bemd` returns a float corresponding to the tail probability
%
$$
B^{\mathrm{EMD}}_{AB} = P(R_A < R_B \mid c) \,.
$$
%
This is the probability that the [expected risk](https://en.wikipedia.org/wiki/Statistical_learning_theory#Formal_description) (aka the expected loss) of model $A$ is less than that of model $B$, _across replications of the experiment_. The fundamental assumption underlying EMD is this:

> Discrepancies between model predictions and observed data are due to uncontrolled experimental factors. Those factors may change across replications.

Computations work by generating for each model a *risk-distribution*, representing our estimate of what the risk would be across experiments. 
Comparisons are tied to the *overlap* between distributions: a model is rejected if it is always worse than another model. For example, if we compute $B^{\mathrm{EMD}}_{AB}$ = 95%, that would mean that in at least 95% of replications, model $A$ would have a lower (better) risk than model $B$. We could then decide to reject model $B$.

The scaling parameter $c$ controls the sensitivity of replications to prediction errors: larger $c$ means more variability, broader risk distributions, increased overlap, and reduced ability to differentiate between models. In our experiments we found that values in the range 2⁻¹ to 2⁻² worked well, but see the paper for a calibration method to identify appropriate for your experiment.
This package provides `emdcmp.tasks.Calibrate` to help run calibration experiments.


Although the single $B^{\mathrm{EMD}}$ number is convenient, much more informative are the risk-distributions themselves. 
These distributions are obtained by sampling, with the function `draw_R_samples`:

```python
RA_samples = emd.draw_R_samples(mixed_ppfA, synth_ppfA, c=c)
RB_samples = emd.draw_R_samples(mixed_ppfB, synth_ppfB, c=c)
```

Samples can then be plotted as a distribution. The example below uses [HoloViews](https://holoviews.org/):
```python
import holoviews as hv
hv.Distribution(RA_samples, label="Model A") \
* hv.Distribution(RB_samples, label="Model B")
```

### Complete usage examples

The documentation contains a [simple example](https://alcrene.github.io/emdcmp/src/emdcmp/emd.html#test-sampling-of-expected-risk-r).
Moreover, all the [code for the paper’s figures] is available, in the form of Jupyter notebooks.
These are heavily commented with extra additional usage hints; they are highly recommended reading.


## Debugging

If computations are taking inordinately long, set the debug level to `DEBUG`:

    
    logging.getLogger("emdcmp").setLevel("DEBUG")

This will print messages to your console reporting how much time each computation step is taking, which should help pin down the source of the issue.

## Further work

The current implementation of the hierarchical beta process (used for sampling quantile paths) has seen quite a lot of testing for numerical stability, but little optimization effort. In particular it makes a lot of calls to functions in `scipy.optimize`, which makes the whole function quite slow: even with a relatively complicated data model like the [pyloric circuit](https://alcrene.github.io/pyloric-network-simulator/pyloric_simulator/prinz2004.html), drawing quantile paths can still take 10x longer than generating the data.

Substantial performance improvements to the sampling algorithm would almost certainly be possible with dedicated computer science effort. This sampling is the main bottleneck, any improvement on this front would also benefit the whole EMD procedure.

The hierarchical beta process is also not the only possible process for stochastically generating quantile paths: it was chosen in part because it made proving integrability particularly simple. Other processes may provide other advantages, either with respect to statistical robustness or computation speed.
