# PyEXAFS

[![Test Status](https://github.com/San-WierPa/pyexafs/workflows/Test/badge.svg?branch=main)](https://github.com/San-WierPa/pyexafs/actions?query=workflow%3ATest)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![imports: isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://github.com/PyCQA/isort)

Developed by Frank Foerste and Sebastian Paripsa, **PyExafs** aims to streamline the workflow<br>
for XAS researchers, from data acquisition to final analysis.

## Introduction

**PyExafs** is designed to facilitate the automated quality assessment<br>
of X-ray Absorption Fine Structure (XAFS) measurements. Leveraging the robust capabilities of the Larch library,<br>
**PyExafs** enables researchers to efficiently process, analyse, and visualise XAFS data, ensuring adherence to stringent quality criteria.

With a single execution, **PyExafs** provides a fast analysis of a dataset, instantly delivering:

+ Absorbance
+ Normalized Absorbance
+ $\chi(k)$
+ $\chi(R)$

These results are presented according to well-defined quality standards, enabling users to quickly evaluate<br>
the integrity of their data.

Key features include:

+ Automated Quality Control: Implements rigorous routines to verify the quality of XAFS measurements.
+ Data Preprocessing: Efficiently handles data loading, preprocessing, and energy calibration.
+ Visualisation: Generates comprehensive plots for raw, normalized, k-space, and R-space data,<br>
supporting both detailed analysis and publication-quality figures.
+ Noise Estimation and Fitting: Provides tools for estimating noise and fitting the first shell,<br>
crucial for accurate data interpretation.

## Installation

Easy:

```sh
python -m pip install pyexafs
```

## Usage

To use the `pyexafs` package, run the following command and provide the path to your data file:

```sh
python -m pyexafs <path_to_data_file>
```

For example:

```sh
python -m pyexafs /path/to/your/datafile.txt
```
