# PyEXAFS

Developed by Frank Foerste and Sebastian Paripsa, **PyExafs** aims to streamline the workflow<br>
for XAFS researchers, from data acquisition to final analysis.

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

You can install PyExafs using pip:

```bash
pip install pyexafs
```
