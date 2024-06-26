#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:33:49 2020

@authors: Frank Foerste and Sebastian Paripsa
ffoerste@physik.tu-berlin.de, paripsa@uni-wuppertal.de
"""

__authors__ = ["Frank Foerste", "Sebastian Paripsa"]
__contact__ = ["ffoerste@physik.tu-berlin.de", "paripsa@uni-wuppertal.de"]
__license__ = "MIT"
__status__ = "production"

##############################################################################
### import packages ###
##############################################################################

import argparse
import base64
import io
import json
import os
from glob import glob
from sys import path, platform
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from larch import Group, Interpreter, fitting, xafs, xray
from PIL import Image
from scipy.signal import argrelextrema

##############################################################################
### Custom imports ###
##############################################################################

path.append("/".join(os.path.abspath(os.curdir).split("/")[:-1]))
from plugins.read_data import ReadData

##############################################################################
### Matplotlib customisations ###
##############################################################################

plt.ioff()
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.right"] = True
plt.rcParams["axes.grid.which"] = "both"
plt.rcParams["font.size"] = 16
plt.rcParams["grid.alpha"] = 0
plt.rcParams["lines.linewidth"] = 1.2
plt.rcParams["lines.markersize"] = 16

##############################################################################
### define quality check and testing feature ###
##############################################################################


class PyExafs(object):
    """
    This class implements the automated routines to check the quality criteria
    of XAFS measurements. The evaluation is based on larch
    https://xraypy.github.io/xraylarch/
    """

    def __init__(self, quality_criteria_json: str, verbose: bool = False) -> None:
        self.verbose = verbose
        if self.verbose:
            print("+++++++++++++++++++++++++++++++++++++++++")
            print("+ quality criteria function initialized +")
            print("+          verbose mode active          +")
            print("+++++++++++++++++++++++++++++++++++++++++")
        with open(quality_criteria_json, "r") as f:
            self.quality_criteria = json.load(f)

    def shorten_data(
        self,
    ) -> None:
        """
        Functionality to cut the last index of the data to avoid unwanted behaviour
        in k
        """
        self.data.energy = self.data.energy[:-1]
        self.data.mu = self.data.mu[:-1]
        self.data.flat = self.data.flat[:-1]
        self.data.pre_edge = self.data.pre_edge[:-1]
        self.data.post_edge = self.data.post_edge[:-1]
        self.data.k = self.data.k[:-1]
        self.data.chi = self.data.chi[:-1]

    def clip_data(
        self, data: np.ndarray, minimum: float = -5, maximum: float = 5
    ) -> np.ndarray:
        """Clip k-data to minimum and maximum to avoid unwanted behaviour in k."""
        return np.clip(data, a_min=minimum, a_max=maximum)

    def load_data(
        self,
        measurement_data: np.ndarray,
        source: str = "SYNCHROTRON",
        mode: str = "ABSORPTION",
        processed: str = "RAW",
        name: Optional[str] = None,
        plot: bool = True,
    ) -> None:
        """
        Load the data to process it.

        Parameters
        ----------
        measurement_data : np.ndarray
            The measurement data, has to be [energy, mu] with same shape.
        source : str, optional
            Type of the utilized x-ray source, either SYNCHROTRON or LABORATORY.
        mode : str, optional
            Mode of measurement, either ABSORPTION or FLUORESCENCE.
        processed : str, optional
            Data processing status, either RAW or PROCESSED.
        name : str, optional
            Name of the sample. If None, defaults to "sample".
        plot : bool, optional
            If True, the data and processing will be plotted.
        """
        if name is None:
            self.name = "sample"
        else:
            self.name = name
        ### store the parameters in the class
        self.plot = plot
        self.source = source
        self.mode = mode
        self.processed = processed
        ### read the correct quality criteria correspondend to the sample
        self.quality_criteria_sample = self.quality_criteria[self.source][self.mode][
            self.processed
        ]
        ### initialize the larch Group to evaluate the data
        self.data = Group()
        ### add energy and absorption to the larch Group
        self.data.energy = measurement_data[0, :]
        self.data.mu = measurement_data[1, :]
        print(measurement_data.shape)

    def preprocess_data(self, take_first: bool = False) -> Group:
        """
        Preprocess the data, find the edge, and fit the pre and post edge.

        Parameters
        ----------
        take_first : boolean, optional
            regarding the edge finding algorithm. if true, the first local
            maximum in the derivative of the measured absorption data is taken
            (as it is done in Athena), if False, the maximum derivative is taken
            (as in Larch). The default is False.
        """
        ### find the edge energy E0 of the absorption data
        self.find_e0(
            self.data.energy, self.data.mu, group=self.data, take_first=take_first
        )
        # self.data.e0 = 7112
        # xafs.find_e0(self.data.energy, self.data.mu, group=self.data)
        ### perform an energy calibration
        ### for this guess the element edge and retrieve the edge energy from
        ### the database of larch mostly based on Elam
        ### https://xraypy.github.io/xraylarch/xray.html
        element_n_edge = xray.guess_edge(self.data.e0)  # ,   energy=7112, edges=['K']
        edge_E_DB = xray.xray_edge(*element_n_edge)[0]
        self.data.E_difference = self.data.e0 - edge_E_DB
        self.data.energy -= self.data.e0 - edge_E_DB
        self.find_e0(
            self.data.energy, self.data.mu, group=self.data, take_first=take_first
        )
        # self.data.e0 = 7112
        ### retrieve the array index of E0 to determine low cut energy
        edge_index = np.where(np.argmin(np.abs(self.data.energy - self.data.e0)))[0][0]
        cut_index = edge_index - 150
        ### if the data below edge is not sufficient, set index to 0 to avoid
        ### using data from the end of the array
        if cut_index < 0:
            cut_index = 0
        if cut_index > edge_index // 2:
            cut_index = (cut_index + edge_index // 2) // 2
        ### cut the energy and absorption in the pre-edge region
        self.data.element_n_edge = element_n_edge
        self.data.energy = self.data.energy[cut_index:]
        self.data.mu = self.data.mu[cut_index:]
        ### calculate the edge position
        xafs.pre_edge(
            energy=self.data.energy,
            mu=self.data.mu,
            e0=self.data.e0,
            group=self.data,
            pre1=-150,
            pre2=-30,
            norm1=50,
            norm2=700,
            make_flat=True,
            nvict=3,
        )
        ### estimate pre and post edge and correct the background
        xafs.autobk(
            energy=self.data.energy,
            mu=self.data.mu,
            group=self.data,
            rbkg=1.0,
            clamp_lo=10,
            clamp_hi=1,
            dk=1.0,
            kweight=2,
            win="hanning",
        )
        ### calculate k**2*chi to determine the k-range for Fourier R transformation
        data = self.data.k**2 * self.data.chi
        ### get root positions to capture whole fluctuation periods
        positive = np.where(
            np.clip(
                np.diff(np.sign(data[(self.data.k > 2) & (self.data.k < 13)])),
                0,
                np.inf,
            )
        )[0]
        negative = np.where(
            np.clip(
                np.diff(np.sign(data[(self.data.k > 2) & (self.data.k < 13)])),
                -np.inf,
                0,
            )
        )[0]
        #### determine kmin and kmax and cap kmax to 15
        self.kmin = self.data.k[(self.data.k > 2) & (self.data.k < 13)][positive[0]]
        self.kmax = self.data.k[(self.data.k > 2) & (self.data.k < 13)][negative[-1]]
        if self.kmax > 13:
            self.kmax = 13
        ### transform data to R
        xafs.xftf(
            k=self.data.k,
            chi=self.data.chi,
            group=self.data,
            dk=1,
            kmin=self.kmin,
            kmax=self.kmax,
            kweight=2,
            rmax_out=12,
            window="hanning",
        )

        ### estimate noise with larch, this is but a estimation and should be
        ### regarded with caution!
        ### TODO
        xafs.estimate_noise(
            k=self.data.k,
            chi=self.data.chi,
            group=self.data,
            rmin=self.data.r.max() - self.data.r.max() * 0.25,
            rmax=self.data.r.max(),
            _larch=Interpreter(),
        )
        return self.data

    def find_e0(
        self,
        energy: np.ndarray,
        mu: np.ndarray,
        group: Optional[Group] = None,
        take_first: bool = False,
    ) -> float:
        """
        Find the edge energy (e0) using the maximal derivative value.

        Parameters
        ----------
        energy : np.ndarray
            Energy of the measurement.
        mu : np.ndarray
            Absorption of the measurement.
        group : larch.group, optional
            Larch group. The default is None.
        take_first : bool, optional
            If True, take the first local derivative maximum (Athena-like),
            otherwise the maximum derivative (Larch-like). The default is False.

        Returns
        -------
        float
            Edge energy e0.
        """
        if len(energy.shape) > 1:
            energy = energy.squeeze()
        if len(mu.shape) > 1:
            mu = mu.squeeze()
        dmu = np.gradient(mu) / np.gradient(energy)
        # find points of high derivative
        dmu[np.where(~np.isfinite(dmu))] = 0
        nmin = max(3, int(len(dmu) * 0.05))
        maxdmu = max(dmu[nmin:-nmin])
        high_deriv_pts = np.where(dmu > maxdmu * 0.1)[0]
        high_deriv_pts = high_deriv_pts[
            dmu[high_deriv_pts] > np.mean(dmu[high_deriv_pts])
        ]
        maxima_indices = argrelextrema(dmu[high_deriv_pts], np.greater)
        # Access the values at the maxima indices
        if take_first:
            e0_idx = np.take(high_deriv_pts[maxima_indices], 0)
            e0 = energy[e0_idx]
        else:
            e0_idx = np.max(high_deriv_pts[maxima_indices])
            e0 = energy[e0_idx]
        if group:
            group.e0 = e0
        return e0

    def plot_data(
        self,
        data_type: str = "RAW",
        show_name: bool = True,
        show: bool = False,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate plots of different data types.

        Parameters
        ----------
        data_type : str, optional
            Type of data to be plotted. Either "RAW", "NORMALIZED", "k", "R" or "BACKGROUND". The default is 'RAW'.
        show_name : bool, optional
            True if the plots shall be plotted in an extra window (for debugging). The default is True.
        show : bool, optional
            True if the plots shall be displayed. The default is False.
        save_path : str, optional
            Absolute path to a folder, where the plots shall be saved. The default is None.

        Returns
        -------
        plt.Figure
            Matplotlib figure with the data plotted.
        """
        ### define figure
        self.fig_data = plt.figure(
            "{} {}".format(data_type, self.name), figsize=(10, 6.25)
        )
        self.fig_data.clf()
        self.ax_data = self.fig_data.subplots()
        self.ax_data.grid()
        ### calculate default ticks
        major_ticks_exafs = np.arange(
            int(np.round(self.data.energy[0], decimals=-1)),
            int(self.data.energy[-1]),
            100,
        )
        minor_ticks_exafs = np.arange(
            int(np.round(self.data.energy[0], decimals=-1)),
            int(self.data.energy[-1]),
            20,
        )
        major_ticks_xanes = np.arange(
            int(np.round(self.data.energy[0], decimals=-1)),
            int(self.data.energy[-1]),
            20,
        )
        minor_ticks_xanes = np.arange(
            int(np.round(self.data.energy[0], decimals=-1)),
            int(self.data.energy[-1]),
            5,
        )
        # major_ticks = np.arange(self.data.energy[0], self.data.energy[-1], 100)
        # minor_ticks = np.arange(self.data.energy[0], self.data.energy[-1], 20)
        ### legend location
        loc = "lower right"
        ### plot data depending on type
        if data_type == "RAW" or data_type == "BACKGROUND":
            ### plotting
            if show_name:
                label = f"Measurement {self.name}"
            else:
                label = "Measurement"
            self.ax_data.plot(
                self.data.energy, self.data.mu, label=label, color="#003161"
            )
            self.ax_data.plot(
                self.data.e0,
                self.data.mu[np.where(self.data.e0 == self.data.energy)],
                marker="*",
                color="#69398B",
                label="Edge Position",
            )
            ### if background shall be plotted
            if data_type == "BACKGROUND":
                if show_name:
                    label = f"Flattened Normalized {self.name}"
                else:
                    label = "Flattened Normalized"
                self.ax_data.plot(
                    self.data.energy, self.data.pre_edge, label="Pre Edge Background"
                )
                self.ax_data.plot(
                    self.data.energy, self.data.post_edge, label="Post Edge Background"
                )
                self.ax_data.plot(
                    self.data.energy,
                    self.data.flat,
                    label=label,
                )
            ### labelling
            self.ax_data.set_xlabel(r"Energy | eV")
            self.ax_data.set_ylabel(r"$\mu (E)$ | a.u.")
            ### set ticks
            self.ax_data.set_xticks(major_ticks_exafs)
            self.ax_data.set_xticks(minor_ticks_exafs, minor=True)
            ### limiting
            self.ax_data.set_xlim(self.data.energy[0], self.data.energy[-1])
        elif data_type == "NORMALIZED":
            ### plotting
            if show_name:
                label = f"{data_type} {self.name}"
            else:
                label = f"{data_type}"
            self.ax_data.plot(
                self.data.energy, self.data.flat, label=label, color="#003161"
            )
            self.ax_data.plot(
                self.data.e0,
                self.data.flat[np.where(self.data.e0 == self.data.energy)],
                marker="*",
                color="#69398B",
                label="Edge Position",
            )
            ### labelling
            self.ax_data.set_xlabel(r"Energy | eV")
            self.ax_data.set_ylabel(r"$\mu (E)$ | a.u.")
            ### set ticks
            self.ax_data.set_xticks(major_ticks_xanes)
            self.ax_data.set_xticks(minor_ticks_xanes, minor=True)
            ### limiting
            self.ax_data.set_xlim(self.data.e0 - 30, self.data.e0 + 100)
            self.ax_data.set_ylim(0)
            # self.ax_data.set_xlim(self.data.energy[0], self.data.energy[-1])
        elif data_type == "k":
            ### calculate specific ticks
            major_ticks = np.arange(self.data.k[0], self.data.k[-1], 2)
            minor_ticks = np.arange(self.data.k[0], self.data.k[-1], 0.5)
            ### calculate data k**2*chi
            data = self.data.k**2 * self.data.chi
            data = self.clip_data(data)
            ### plotting
            self.ax_data.plot(self.data.k, data, label=f"{data_type}", color="#003161")
            ### labelling
            self.ax_data.set_xlabel(r"k | $\AA^{-1}$")
            self.ax_data.set_ylabel(r"$k^2\chi(k) | \AA^{-1}$")
            ### set ticks
            self.ax_data.set_xticks(major_ticks)
            self.ax_data.set_xticks(minor_ticks, minor=True)
            ### limiting
            if self.data.k.max() > 15:
                xmax = 15
            else:
                xmax = self.data.k.max()
            # self.ax_data.set_xlim(self.data.k[0], self.data.k[-1])
            self.ax_data.set_xlim(0, xmax)
            ymin = np.min(data[len(data) // 8 : -len(data) // 8]) - 0.01
            ymax = np.max(data[len(data) // 8 : -len(data) // 8]) + 0.01
            # if ymin < -3: ymin = -3
            # if ymax > 3: ymax = 3
            self.ax_data.set_ylim(ymin, ymax)
            loc = "upper left"
        elif data_type == "R":
            ### calculate specific ticks
            major_ticks = np.arange(self.data.r[0], self.data.r[-1], 2)
            minor_ticks = np.arange(self.data.r[0], self.data.r[-1], 0.5)
            ### plotting
            self.ax_data.plot(
                self.data.r,
                np.abs(self.data.chir),
                label="{}".format(data_type),
                color="#003161",
            )
            ### labelling
            self.ax_data.set_xlabel(r"$R(\AA)$")
            self.ax_data.set_ylabel(r"$\left| \chi(R) \right| \AA^{-3}$")
            ### set ticks
            self.ax_data.set_xticks(major_ticks)
            self.ax_data.set_xticks(minor_ticks, minor=True)
            ### limiting
            self.ax_data.set_xlim(0, 6)
            self.ax_data.set_ylim(0)
            ### legend positioning
            loc = "upper right"
            # self.ax_data.set_xlim(self.data.r[0], self.data.r[-1])
        ### set title
        if show_name:
            self.ax_data.set_title(self.name)
        ### set legend
        self.ax_data.legend(loc=loc)
        ### show figure if desired
        if show:
            self.fig_data.show()
        if save_path:
            self.fig_data.savefig(save_path)
        return self.fig_data

    def check_edge_step(self) -> Tuple[bool, float]:
        """
        Evaluate the edge step of the given data.

        Returns
        -------
        Tuple[bool, float]
            A tuple containing a boolean indicating if the edge step meets the quality criteria
            and the value of the edge step.
        """
        if (
            self.quality_criteria_sample["edge step"]["min"]
            <= self.data.edge_step
            <= self.quality_criteria_sample["edge step"]["max"]
        ):
            if self.verbose:
                print(
                    "\u2705 edge step of good quality: {:.2f}".format(
                        self.data.edge_step
                    )
                )
            return True, self.data.edge_step
        else:
            if self.verbose:
                print(
                    "\u274e edge step doesn't meet standards: {:.2f}".format(
                        self.data.edge_step
                    )
                )
            return False, self.data.edge_step

    def check_energy_resolution(self) -> Tuple[bool, float]:
        """
        Evaluate the energy resolution of the given data.

        Returns
        -------
        Tuple[bool, float]
            A tuple containing a boolean indicating if the energy resolution meets the quality criteria
            and the value of the energy resolution.
        """
        self.data.energy_resolution = self.data.energy[1] - self.data.energy[0]
        if (
            self.quality_criteria_sample["energy resolution"]["min"]
            <= self.data.energy_resolution
            <= self.quality_criteria_sample["energy resolution"]["max"]
        ):
            if self.verbose:
                print(
                    "\u2705 energy resolution of good quality: {:.2f}eV".format(
                        self.data.energy_resolution
                    )
                )
            return True, self.data.energy_resolution
        else:
            if self.verbose:
                print(
                    "\u274e energy resolution doesn't meet standards: {:.2f}eV".format(
                        self.data.energy_resolution
                    )
                )
            return False, self.data.energy_resolution

    def check_k(self) -> Tuple[bool, float]:
        """
        Evaluate the k range of the given data.

        Returns
        -------
        Tuple[bool, float]
            A tuple containing a boolean indicating if the k range meets the quality criteria
            and the value of k max.
        """
        if (
            self.quality_criteria_sample["k max"]["min"]
            <= self.data.k[-1]
            <= self.quality_criteria_sample["k max"]["max"]
        ):
            if self.verbose:
                print(
                    "\u2705 k max of good quality: {:.2f}\u212b⁻¹".format(
                        self.data.k[-1]
                    )
                )
            return True, self.data.k[-1]
        else:
            if self.verbose:
                print(
                    "\u274e k max doesn't meet standards: {:.2f}\u212b⁻¹".format(
                        self.data.k[-1]
                    )
                )
            return False, self.data.k[-1]

    def estimate_noise(
        self,
    ):
        """
        this function automatically estimates the noise in the k regime of
        the given data
        """

        if (
            self.quality_criteria_sample["noise"]["min"]
            <= self.data.epsilon_k
            <= self.quality_criteria_sample["noise"]["max"]
        ):
            if self.verbose:
                print(
                    "\u2705 estimated noise of good quality: {:.2f}".format(
                        self.data.epsilon_k
                    )
                )
            return True, self.data.epsilon_k
        else:
            if self.verbose:
                print(
                    "\u274e estimated noise doesn't meet standards: {:.2f}".format(
                        self.data.epsilon_k
                    )
                )
            return False, self.data.epsilon_k

    def first_shell_fit(
        self,
    ):
        """
        Function to automatically fit the first shell with larch. Not yet
        implemented #TODO
        """
        pars = fitting.param_group(
            amp=fitting.param(1.0, vary=True),
            del_e0=fitting.param(0.0, vary=True),
            sig2=fitting.param(0.0, vary=True),
            del_r=fitting.guess(0.0, vary=True),
        )

    def encode_base64_figure(self, figure):
        """
        encode a matplotlib figure to base64 for storage

        Parameters
        ----------
        figure : matplotlib.figure
            matplotlib figure with relevant data.

        Returns
        -------
        str
            base64 encoded string of the figure.

        """
        buffer = io.BytesIO()
        figure.savefig(buffer, format="jpeg")
        data = base64.b64encode(buffer.getbuffer()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"

    def decode_base64_figure(self, base64_string):
        """
        decode a matplotlib figure from base64

        Parameters
        ----------
        base64_string : str
            base64 encoded matplotlib.figure

        Returns
        -------
        image
            image of the matplotlib.figure

        """
        image_base64 = base64_string.replace("data:image/jpeg;base64,", "")
        image_base64 = base64.b64decode(image_base64)
        image_data = io.BytesIO(image_base64)
        image = Image.open(image_data)
        return image


class PyExafsControl(object):
    """
    This checks the quality control for a given facility type. It looks for data
    in the example data folder.
    Supported facility_type are LABORATORY; SYNCHROTRON and must be a string.
    The plot_ variables have to be BOOLEAN. They define, if a dataset is plotted
    or not.
    """

    def __init__(
        self,
        facility_type,
        files=None,
        plot_raw_data=False,
        plot_normalized_data=False,
        plot_k=False,
        plot_R=False,
        plot_background=False,
        save_figure_path=None,
        take_first=False,
        verbose=False,
    ):
        """
        Parameters
        ----------
        facility_type : str
            type of the facility, either SYNCHROTRON or LABORATORY.
        files : list, optional
            list of the files to check the quality, if None given the default
            files are evaluated
        plot_raw_data : bool, optional
            plot µ(E)_measured. The default is False.
        plot_normalized_data : bool, optional
            plot µ(E)_normalized. The default is False.
        plot_k : bool, optional
            plot chi(k). The default is False.
        plot_R : bool, optional
            plot chi(R). The default is False.
        save_figure_path : str, optional
            absolute path to the folder where figures shall be stored.
            The default is None.
        take_first: boolean, optional
            find_e0: wether to take the first or maximum derivative (Athena vs Larch)
        verbose : bool, optional
            if True certain data are printed
        """
        ### store the given data in the self instance
        self.facility_type = facility_type
        self.files = files
        self.plot_raw_data = plot_raw_data
        self.plot_normalized_data = plot_normalized_data
        self.plot_k = plot_k
        self.plot_R = plot_R
        self.plot_background = plot_background
        self.save_figure_path = save_figure_path
        self.take_first = take_first
        self.verbose = verbose
        ### initialize the read_data plugin with the facility type
        ### !!! only one type allowed per init
        self.read_data = ReadData(source=facility_type)
        ### perform the quality control
        self.results = self.check_data()

    def compare_plotting(self, data, data_type, save_path=None):
        """
        Function to plot comparison data in the same figure
        data_type : str
            type of data, RAW, k, R, NORMALIZED
        """
        if data_type == "RAW":
            print(data_type, dir(data))
            self.cq.ax_data.plot(
                data.ee, data.xmu, label="Athena", linestyle="dashed", color="red"
            )
        elif data_type == "NORMALIZED":
            print(data_type, dir(data))
            self.cq.ax_data.plot(
                data.energy, data.norm, label="Athena", linestyle="dashed", color="red"
            )
            # self.cq.ax_data.plot(data.energy, data.der_norm, label='derivative',
            #                      linestyle='dashed', color='blue')
        elif data_type == "R":
            print(data_type, dir(data))
            self.cq.ax_data.plot(
                data.r, data.chir_mag, label="Athena", linestyle="dashed", color="red"
            )
            self.cq.ax_data.autoscale(enable=True, axis="y", tight=True)
        elif data_type == "k":
            print(data_type, dir(data))
            self.cq.ax_data.plot(
                data.k, data.chik, label="Athena", linestyle="dashed", color="red"
            )
            self.cq.ax_data.axhline(y=0, color="black", lw=0.8)
        self.cq.ax_data.legend()
        self.cq.fig_data.tight_layout()
        if save_path:
            self.cq.fig_data.savefig(save_path + f"{self.name}_{data_type}.png")

    def check_data(
        self,
    ):
        """
        This function reads out the quality criteria from the Criteria.json,
        search for all files in the specific examples data folder and checks
        the quality for each sample iterative. If verbose mode is activated
        the results are printed.

        """
        ### read out quality criteria
        cq_json = "src/pyexafs/criteria.json"
        ### check out all files
        if self.files is None:
            folder = "/src/pyexafs/example_data/SYNCHROTRON/"
            files = sorted(glob(folder + "*"))[1:2]
        else:
            files = self.files
        ### initialize the PyExafs class
        self.cq = PyExafs(quality_criteria_json=cq_json, verbose=self.verbose)
        ### analyse the quality for each file in the files list
        for file in files:
            if self.verbose:
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("working on {}".format(file.split("/")[-1]))
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("file:\t", file.split("/")[-1])
            ### transform the name variable corresponding to the host platform
            if "win" in platform:
                self.name = file.split("\\")[-1].split(".")[0]
            else:
                self.name = file.split("/")[-1].split(".")[0]
            ### initialize the quality control list to store quality data of
            ### the analysed file
            self.qc_list = []
            ### read out the data of the file
            self.read_data.process_data(data_path=file)
            self.cq.load_data(
                self.read_data.data, source=self.facility_type, name=self.name
            )
            self.data = self.cq.preprocess_data(take_first=self.take_first)
            if self.verbose:
                print("meas data loaded:", self.read_data.data.shape)
                print("guessed element and edge: ", self.cq.data.element_n_edge)
                print("E0: {:.0f}eV".format(self.cq.data.e0))
                print("k-range: {:.1f}-{:.1f}".format(self.cq.kmin, self.cq.kmax))
            if self.save_figure_path:
                show = False
            else:
                show = True
                save_path = None
            if self.plot_raw_data:
                if self.save_figure_path:
                    save_path = self.save_figure_path + "/RAW/{}_RAW.png".format(
                        self.name
                    )
                self.fig_raw_data = self.cq.plot_data(
                    data_type="RAW", show_name=False, show=show, save_path=save_path
                )
                self.fig_raw_data_base64 = self.cq.encode_base64_figure(
                    self.fig_raw_data
                )
                image_data = self.cq.decode_base64_figure(
                    base64_string=self.fig_raw_data_base64
                )
            if self.plot_normalized_data:
                if self.save_figure_path:
                    save_path = (
                        self.save_figure_path
                        + "/NORMALIZED/{}_NORMALIZED.png".format(self.name)
                    )
                self.fig_normalized_data = self.cq.plot_data(
                    data_type="NORMALIZED",
                    show_name=False,
                    show=show,
                    save_path=save_path,
                )
                self.fig_normalized_data_base64 = self.cq.encode_base64_figure(
                    self.fig_normalized_data
                )
                image_data = self.cq.decode_base64_figure(
                    base64_string=self.fig_normalized_data_base64
                )
            if self.plot_k:
                if self.save_figure_path:
                    save_path = self.save_figure_path + "/k/{}_k.png".format(self.ame)
                self.fig_k = self.cq.plot_data(
                    data_type="k", show_name=False, show=show, save_path=save_path
                )
                self.fig_k_base64 = self.cq.encode_base64_figure(self.fig_k)
                image_k = self.cq.decode_base64_figure(base64_string=self.fig_k_base64)
            if self.plot_R:
                if self.save_figure_path:
                    save_path = self.save_figure_path + "/R/{}_R.png".format(self.name)
                self.fig_R = self.cq.plot_data(
                    data_type="R", show_name=False, show=show, save_path=save_path
                )
                self.fig_R_base64 = self.cq.encode_base64_figure(self.fig_R)
                image_R = self.cq.decode_base64_figure(base64_string=self.fig_R_base64)
            if self.plot_background:
                if self.save_figure_path:
                    save_path = (
                        self.save_figure_path
                        + "/BACKGROUND/{}_BACKGROUND.png".format(self.name)
                    )
                self.fig_background = self.cq.plot_data(
                    data_type="BACKGROUND",
                    show_name=False,
                    show=show,
                    save_path=save_path,
                )
                self.fig_background_base64 = self.cq.encode_base64_figure(
                    self.fig_background
                )
                image_background = self.cq.decode_base64_figure(
                    base64_string=self.fig_background_base64
                )

            self.qc_list.append(self.cq.check_edge_step())
            self.qc_list.append(self.cq.check_energy_resolution())
            self.qc_list.append(self.cq.check_k())
            self.qc_list.append(self.cq.estimate_noise())

            if self.verbose:
                if all(np.array(self.qc_list)[:, 0]):
                    print("quality approved")
                else:
                    print("data not matchs all quality criteria, please check")
            self.cq.first_shell_fit()
        return self.cq.data


def main():
    """
    Main function to execute the quality control and plotting.
    """
    parser = argparse.ArgumentParser(
        description="Run EXAFS data quality control and plotting."
    )
    parser.add_argument("file", type=str, help="Path to the data file")

    args = parser.parse_args()
    file_path = args.file

    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    test = PyExafsControl(
        facility_type="SYNCHROTRON",
        files=[file_path],
        take_first=False,
    )
    test.cq.plot_data("RAW", show=True)
    test.cq.plot_data("NORMALIZED", show=True)
    test.cq.plot_data("k", show=True)
    test.cq.plot_data("R", show=True)
    plt.show()


if __name__ == "__main__":
    main()
