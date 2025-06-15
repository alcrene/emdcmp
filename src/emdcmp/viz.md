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

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
---
from __future__ import annotations
```

```{code-cell}
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import holoviews as hv
```

```{code-cell}
from dataclasses import dataclass
from typing import Optional, Dict, List
```

```{code-cell}
---
editable: true
raw_mimetype: ''
slideshow:
  slide_type: ''
tags: [skip-execution]
---
from .config import config
from . import utils
```

```{code-cell}
---
editable: true
slideshow:
  slide_type: ''
tags: [active-ipynb, remove-cell]
---
from config import config
import utils
```

```{code-cell}
dash_patterns = ["dotted", "dashed", "solid"]
```

# Plotting functions

```{code-cell}
@dataclass
class CalibrationPlotElements:
    """
    `bin_idcs`: Dictionary indicating which experiment index were assigned to
       each bin. Use in conjunction with the EpistemicDistribution iterator
       to reconstruct specific experiments.
    """
    calibration_curves: hv.Overlay
    prohibited_areas   : hv.Area
    discouraged_areas : hv.Overlay
    bin_idcs: Dict[float,List[np.ndarray[int]]]

    def __iter__(self):
        yield self.calibration_curves
        yield self.prohibited_areas
        yield self.discouraged_areas

    def _repr_mimebundle_(self, *args, **kwds):
        return (self.prohibited_areas * self.discouraged_areas * self.calibration_curves)._repr_mimebundle_(*args, **kwds)
```

```{code-cell}
def calibration_bins(calib_results: CalibrateResult,
                     target_bin_size: Optional[int]=None):
    """Return the bin edges for the histograms produced by `calibration_plot`.
    
    .. Note:: These are generally *not* good bin edges for plotting a histogram
    of calibration results: by design, they will produce an almost
    flat histogram.
    """
    bin_edges = {}
    for c, data in calib_results.items():
        i = 0
        Bemd = np.sort(data["Bemd"])
        edges = [Bemd[0]]
        for w in utils.get_bin_sizes(len(Bemd), target_bin_size)[:-1]:
            i += w
            edges.append(Bemd[i:i+2].mean())
        edges.append(Bemd[-1])
        bin_edges[c] = edges
    return bin_edges
```

```{code-cell}
def calibration_plot(calib_results: CalibrateResult,
                     target_bin_size: Optional[int]=None
                    ) -> CalibrationPlotElements:
    """Create a calibration plot from the results of calibration experiments.

    Parameters
    ----------
    calib_results: The calibration results to plot. The typical way to obtain
       these is to create and run `Calibrate` task:
       >>> task = emdcmp.tasks.Calibrate(...)
       >>> calib_results = task.unpack_results(task.run())
    target_bin_size: Each point on the calibration curve is an average over
       some number of calibration experiments; this parameter sets that number.
       (The actual number may vary a bit, if `target_bin_size` does not exactly
       divide the total number of samples.)
       Larger bin sizes result in fewer but more accurate curve points.
       The default is to aim for the largest bin size possible which results
       in 16 curve points, with some limits in case the number of results is
       very small or very large.
    """

    ## Calibration curves ##
    calib_curves = {}
    calib_bins = {}
    for c, data in calib_results.items():
        # # We don’t do the following because it uses the Bepis data to break ties.
        # # If there are a lot equal values (typically happens with a too small c),
        # # then those will get sorted and we get an artificial jump from 0 to 1
        # data.sort(order="Bemd")
        σ = np.argsort(data["Bemd"])  # This will only use Bemd data; order within ties remains random
        Bemd = data["Bemd"][σ]    # NB: Don’t modify original data order: 
        Bepis = data["Bepis"][σ]  #     we may want to inspect it later.

        curve_data = []
        bin_idcs = []
        i = 0
        for w in utils.get_bin_sizes(len(data), target_bin_size):
            curve_data.append((Bemd[i:i+w].mean(),
                               Bepis[i:i+w].mean()))
            bin_idcs.append(σ[i:i+w])
            i += w

        curve = hv.Curve(curve_data, kdims="Bemd", vdims="Bepis", label=f"{c=}")
        curve = curve.redim.range(Bemd=(0,1), Bepis=(0,1))
        curve.opts(hv.opts.Curve(**config.viz.calibration_curves))
        calib_curves[c] = curve
        calib_bins[c] = bin_idcs

    calib_hmap = hv.HoloMap(calib_curves, kdims=["c"])

    ## Prohibited & discouraged areas ##
    # Prohibited area
    prohibited_areas = hv.Area([(x, x, 1-x) for x in np.linspace(0, 1, 32)],
                              kdims=["Bemd"], vdims=["Bepis", "Bepis2"],
                              group="overconfident area")

    # Discouraged areas
    discouraged_area_1 = hv.Area([(x, 1-x, 1) for x in np.linspace(0, 0.5, 16)],
                         kdims=["Bemd"], vdims=["Bepis", "Bepis2"],
                         group="undershoot area")
    discouraged_area_2 = hv.Area([(x, 0, 1-x) for x in np.linspace(0.5, 1, 16)],
                         kdims=["Bemd"], vdims=["Bepis", "Bepis2"],
                         group="undershoot area")

    prohibited_areas = prohibited_areas.redim.range(Bemd=(0,1), Bepis=(0,1))
    discouraged_area_1 = discouraged_area_1.redim.range(Bemd=(0,1), Bepis=(0,1))
    discouraged_area_2 = discouraged_area_2.redim.range(Bemd=(0,1), Bepis=(0,1))

    prohibited_areas.opts(hv.opts.Area(**config.viz.prohibited_area))
    discouraged_area_1.opts(hv.opts.Area(**config.viz.discouraged_area))
    discouraged_area_2.opts(hv.opts.Area(**config.viz.discouraged_area))

    ## Combine & return ##
    return CalibrationPlotElements(
        calib_hmap, prohibited_areas, discouraged_area_1*discouraged_area_2,
        calib_bins)
```
