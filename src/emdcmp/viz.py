# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,md:myst
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (emdcmp-dev)
#     language: python
#     name: emdcmp-dev
# ---

# %% editable=true slideshow={"slide_type": ""}
from __future__ import annotations

# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import holoviews as hv

# %%
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# %% editable=true raw_mimetype="" slideshow={"slide_type": ""} tags=["skip-execution"]
from .config import config
from . import utils

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "remove-cell"]
# from config import config
# import utils

# %%
dash_patterns = ["dotted", "dashed", "solid"]


# %% [markdown]
# # Plotting functions

# %%
@dataclass
class CalibrationPlotElements:
    """
    This object contains summarized data from a calibration experiment, and
    allows to plot it in various forms.
    While it already renders itself graphically as a configuration curve
    with default paramaters, it also provides various options to customize
    the display.

    For plotting calibration curves, use one of the following properties:

    - ``scatters``
    - ``lines``
    - ``overlayed_scatters``
    - ``overlayed_lines``

    The first two return multiple curves (indexed by their *c* value) as a
    `hv.HoloMap`, whereas the latter two combine them into one figure with
    `hv.Overlay`.

    All of these combine three plot elements into an Overlay plot.
    If you want full contral, you can also be retrieve them individually:

    - ``calibration_curves``
    - ``prohibited_areas``
    - ``discouraged_areas``

    Generally the `scatters` plots are recommended, as they better show how
    the bins are distributed, although `lines` can work better for busy plots.

    An alternative way of visualizing how bins are distributed is to plot
    them as histograms with one of:

    - ``Bemd_hists``
    - ``Bepis_hists``

    Since this returns Holoviews plots, the visual style can be adjusted post-hoc
    with the `.opts` method on the returned plot object.
    Alternatively, you may subclass this dataclass and override one or more
    of its plot options methods:

    - ``opts``
    - ``scatter_opts``
    - ``hist_opts``

    Converting a plain CalibPlotElements dataclass to your customized type
    should be as simple as ``MyCalibPlot(old_calib_plot)``.

    Finally, in some cases you may want to reconstruct which experiments were
    assigned to each bin. This can be done with

    - ``experiment_idcs``

    This is a dictionary in the form ``{c: [Array[experiment idx]]]}``.
    It maps each *c* value to a list of arrays, each array containing all
    experiment indices combined into one Bemd bin.
    By re-running the EpistemicDist generator and matching experiment indices,
    it is possible to reconstruct the parameters of each experiment which
    landed in a given bin.
    """
    calibration_curves: hv.HoloMap
    prohibited_areas   : hv.Area
    discouraged_areas : hv.Overlay
    Bemd_hist_data:  Dict[float,Tuple[np.ndarray[int], np.ndarray[float]]]
    Bepis_hist_data: Dict[float,Tuple[np.ndarray[int], np.ndarray[float]]]
    experiment_idcs: Dict[float,List[np.ndarray[int]]]

    ## Expose certain methods so that contained elements can be operated on as a block

    def select(self, selection_specs=None, **kwargs) -> "CalibrationPlotElements":
        """
        Create a new `CalibrationPlotElements` after applying `select` to the calibration curves.
        This is a convenient way of reducing the set of plotted curves.

        Note
        ----
        Other plot elements (the shaded areas) are not cloned, and thus will
        share options and dimensions with the originals.
        """
        return CalibrationPlotElements(
            calibration_curves = self.calibration_curves.select(selection_specs, **kwargs),
            prohibited_areas = self.prohibited_areas,
            discouraged_areas = self.discouraged_areas,
            bin_idcs = self.bin_idcs,
            Bemd_hist_data = self.Bemd_hist_data,
            Bepis_hist_data = self.Bepis_hist_data
        )

    def clone(self, shared_data=True, link=True) -> "CalibrationPlotElements":
        """
        Create a new `CalibrationPlotElements` by cloning all the contained elements.
        Arguments are passed down to all `.clone()` calls

        Note
        ----
        Only `shared_data` and `link` arguments are supported, since other
        arguments don’t make sense for a batch operation.
        """
        return CalibrationPlotElements(
            calibration_curves = self.calibration_curves.clone(shared_data=shared_data, link=link),
            prohibited_areas = self.prohibited_areas.clone(shared_data=shared_data, link=link),
            discouraged_areas = self.discouraged_areas.clone(shared_data=shared_data, link=link),
            bin_idcs = self.bin_idcs,
            Bemd_hist_data = self.Bemd_hist_data,
            Bepis_hist_data = self.Bepis_hist_data
        )

    def redim(self, **kwargs) -> "CalibrationPlotElements":
        """
        Create a new `CalibrationPlotElements` by calling `redim` all the contained elements.
        """
        return CalibrationPlotElements(
            calibration_curves = self.calibration_curves.redim(**kwargs),
            prohibited_areas = self.prohibited_areas.redim(**kwargs),
            discouraged_areas = self.discouraged_areas.redim(**kwargs),
            bin_idcs = self.bin_idcs,
            Bemd_hist_data = self.Bemd_hist_data,
            Bepis_hist_data = self.Bepis_hist_data
        )

    ## Plotting functions ##

    # NB: Using cached_properties allows to overwrite colours on a per-object basis
    @cached_property
    def scatter_palette(self): return config.viz.calibration_curves["color"]
    @cached_property
    def curve_palette(self): return config.viz.calibration_curves["color"]

    @property
    def opts(self):
        return (
            hv.opts.Curve(**config.viz.calibration_curves),
            hv.opts.Area("Overconfident_area", **config.viz.prohibited_area),
            hv.opts.Area("Undershoot_area", **config.viz.discouraged_area)
            )
    @property
    def scatter_opts(self):
        scatter_opts = [hv.opts.Curve(color="#888888"),
                        hv.opts.Scatter(color=self.scatter_palette)
                        ]
        if "matplotlib" in hv.Store.renderers:
            scatter_opts += [hv.opts.Curve(linestyle="dotted", linewidth=1, backend="matplotlib"),
                             hv.opts.Scatter(s=10, backend="matplotlib")]
        if "bokeh" in hv.Store.renderers:
            scatter_opts += [hv.opts.Curve(line_dash="dotted", line_width=1, backend="bokeh")]
        return scatter_opts
    @property
    def hist_opts(self):
        hist_opts = []
        if "matplotlib" in hv.Store.renderers:
            hist_opts.append(
                hv.opts.Histogram(backend="bokeh",
                    line_color=None, alpha=0.75,
                    color=self.curve_palette)
                )
        if "bokeh" in hv.Store.renderers:
            hist_opts.append(
                hv.opts.Histogram(backend="matplotlib",
                    color="none", edgecolor="none", alpha=0.75,
                    facecolor=self.curve_palette)
                )
        return hist_opts

    @property
    def lines(self) -> hv.HoloMap:
        """
        Plot the calibration curves as solid lines joining the (Bemd, Bepis) tuples.
        """
        # We use .clone() to prevent contamination with different view options
        # It must be applied to the Curves themselves; cloning their containing HoloMap is not sufficient
        return hv.HoloMap({c: self.prohibited_areas * self.discouraged_areas
                              * curve.clone()
                           for c, curve in self.calibration_curves.items()},
                           kdims=["c"],
                           sort=False  # Keep the same order as calibration_curves, to ensure consistent legends
               ).opts(*self.opts)

    @property
    def overlayed_lines(self) -> hv.Overlay:
        """
        Plot the calibration curves as solid lines joining the (Bemd, Bepis) tuples.
        """
        # We use .clone() to prevent contamination with different view options
        # It must be applied to the Curves themselves; cloning their containing HoloMap is not sufficient
        return (self.prohibited_areas * self.discouraged_areas
                * hv.Overlay([curve.clone() for curve in self.calibration_curves])
               ).opts(*self.opts)


    @property
    def scatters(self) -> hv.HoloMap:
        """
        Plot the calibration (Bemd, Bepis) tuples as a scatter plot.
        Points are joined with grey dotted lines to allow to make it easier to
        see which points come from the same `c` value and what curve they form.

        When possible, this should be the preferred way of reporting calibration
        curves: by showing where the bins fall, it is much easier to identify
        an excess concentration of points on the edge.
        With many curves however, they are easier to differentiate with the
        solid `lines` format.
        """
        scatters = {c: curve.to.scatter() for c, curve in self.calibration_curves.items()}
        return hv.HoloMap({c: self.prohibited_areas * self.discouraged_areas
                              * self.calibration_curves[c].clone().relabel(label="")  # Remove labels on curves so the legend uses scatter
                              * scatters[c]
                           for c in scatters},
                          kdims=["c"],
                          sort=False  # Keep the same order as calibration_curves, to ensure consistent legends
               ).opts(*self.opts).opts(scatter_opts)
    @property
    def overlayed_scatters(self) -> hv.Overlay:
        """
        Same as `scatters`, except all curve+scatter plots are overlayed
        into one figure.
        """
        scatters = {c: curve.to.scatter() for c, curve in self.calibration_curves.items()}
        return (self.prohibited_areas * self.discouraged_areas
                * hv.Overlay([curve.clone().relabel(label="")  # Remove labels on lines so the legend uses scatter labels (lines all have the same colour, so a legend of lines is no good)
                              for curve in self.calibration_curves.values()])
                * hv.Overlay(list(scatters.values()))
                ).opts(*self.opts).opts(scatter_opts)

    @property
    def Bemd_hists(self) -> hv.HoloMap:
        frames = {c: hv.Histogram(data, kdims=["Bemd"], label="Bemd")
                  for c, data in self.Bemd_hist_data.items()}
        return hv.HoloMap(frames, kdims=["c"], group="Bemd_hists").opts(*self.hist_opts)

    @property
    def Bepis_hists(self) -> hv.HoloMap:
        frames = {c: hv.Histogram(data, kdims=["Bepis"], label="Bepis")
                  for c, data in self.Bepis_hist_data.items()}
        return hv.HoloMap(frames, kdims=["c"], group="Bepis_hists").opts(*self.hist_opts)

    def _repr_mimebundle_(self, *args, **kwds):
        return self.scatters._repr_mimebundle_(*args, **kwds)


# %%
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


# %%
def calibration_hists(calib_results: CalibrateResult,
                      target_bin_size: Optional[int]=None
                    ) -> CalibrationPlotElements:
    """
    Convert Calibration Results into the (Bemd, Bepis) pairs needed for
    calibration plots.
    Recall that on any one experiment, Bepis is either True or False. So to
    estimate the probability P(E[R_A] < E[R_B] | Bemd), we histogram the data
    points into equal-sized bins according to Bemd, then average the value of
    Bepis within each bin. Representing each bin by its midpoint Bemd then
    produces the desired list of (Bemd, Bepis) pairs.
    Note that bins are equal in the number of experiments (and so have the
    equal statistical power), rather than equal in width. The resulting points
    are therefore not equally spaced along the Bemd axis, but will concentrate
    in locations where there are more data.

    When designing calibration experiments, it is important to ensure that there
    are ambiguous cases which probe the middle of the calibration plot – the part
    we actually care about. Otherwise we can end up with all points being
    concentrated in the top right and bottom left corners: while this shows
    a strong correlated, it does not actually tell us whether the probability
    assigned by Bemd is any good, because only clear-cut cases were considered.

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

    Returns
    -------
    curve_data: {c: [(Bemd, Bepis), ...]}
        Dictionary where each entry is a list of data points defining a
        calibration curve for a different c value.
    experiment_ids: {c: [[int,...], ...]}
        Lists of experiment indices used in each Bemd bin.
    """
    ## 
    curve_data = {}
    experiment_idcs = {}
    for c, data in calib_results.items():
        # # We don’t do the following because it uses the Bepis data to break ties.
        # # If there are a lot equal values (typically happens with a too small c),
        # # then those will get sorted and we get an artificial jump from 0 to 1
        # data.sort(order="Bemd")
        σ = np.argsort(data["Bemd"])  # This will only use Bemd data; order within ties remains random
        Bemd = data["Bemd"][σ]        # NB: Don’t modify original data order: 
        Bepis = data["Bepis"][σ]      #     we may want to inspect it later.

        curve_points = []
        bin_idcs = []
        i = 0
        for w in utils.get_bin_sizes(len(data), target_bin_size):
            curve_points.append((Bemd[i:i+w].mean(),
                                 Bepis[i:i+w].mean()))
            bin_idcs.append(σ[i:i+w])
            i += w
        curve_data[c] = curve_points
        experiment_idcs[c] = bin_idcs

    return curve_data, experiment_idcs

# %%
def calibration_plot(calib_results: CalibrateResult,
                     target_bin_size: Optional[int]=None
                    ) -> CalibrationPlotElements:
    """Create a calibration plot from the results of calibration experiments.
    Calls `calibration_hists` to compute the plot data

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

    See also
    --------
    - `calibration_hists`
    """

    ## Calibration curves ##
    curve_data, experiment_idcs = calibration_hists(calib_results, target_bin_size)

    calib_curves = {}
    for c, points in curve_data.items():
        curve = hv.Curve(points, kdims="Bemd", vdims="Bepis", label=f"{c=}")
        curve = curve.redim.range(Bemd=(0,1), Bepis=(0,1))
        # curve.opts(hv.opts.Curve(**config.viz.calibration_curves))
        calib_curves[c] = curve
    calib_hmap = hv.HoloMap(calib_curves, kdims=["c"])

    ## Precompute histograms, so that we can have .*_hist methods to CalibratePlotElements ##
    Bemd_hists = {}
    Bepis_hists = {}
    for c, res in calib_results.items():
        Bemd_hists[c]  = np.histogram(res["Bemd"],              bins="auto", density=False)
        Bepis_hists[c] = np.histogram(res["Bepis"].astype(int), bins="auto", density=False)

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

    # prohibited_areas.opts(hv.opts.Area(**config.viz.prohibited_area))
    # discouraged_area_1.opts(hv.opts.Area(**config.viz.discouraged_area))
    # discouraged_area_2.opts(hv.opts.Area(**config.viz.discouraged_area))

    ## Combine & return ##
    return CalibrationPlotElements(
        calib_hmap, prohibited_areas, discouraged_area_1*discouraged_area_2,
        experiment_idcs, Bemd_hists, Bepis_hists)

