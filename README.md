# Scripts

This repository collects personal scripts, helper files, and analysis tools
used to prepare, run, and process computational chemistry and molecular
dynamics workflows, especially on the CECAR cluster.

Many of the standalone routines are tied to my own day-to-day workflows around
LIO, ORCA, SANDER, SHARC, and related electronic-structure or molecular
dynamics programs. They are public and open to use, but they were written for
personal use first and are not guaranteed to work generically on every machine
or for every file layout.

Use them at your own discretion.

## General Organization

The repository is split into two main areas:

- `personal_scripts/`: personal workflow scripts and legacy utilities.
- `TolkienTools/`: a more general interactive macro-program that groups larger
  analysis routines under a common menu.
- `tolkien-tools`: command-line entry point for `TolkienTools`.

The `personal_scripts/` folder contains scripts for specific workflows:

- `personal_scripts/cecar/`: routines and files intended for running jobs on
  the CECAR cluster.
- `personal_scripts/lio/`: scripts related to LIO calculations and outputs.
- `personal_scripts/orca/`: scripts related to ORCA calculations and outputs.
- `personal_scripts/sharc/`: tools related to SHARC workflows.
- `personal_scripts/procesado/`: scripts and utilities for post-processing results.

## Tolkien Tools

`TolkienTools` is the most general part of this repository. It provides a
terminal menu for running larger analysis routines from a common entry point:

```bash
tolkien-tools
```

Modules can also be launched directly:

```bash
tolkien-tools 1   # TD-DFT spectra
tolkien-tools 2   # charge and spin analysis
tolkien-tools 3   # multilambda kinetics
```

### Main Modules

1. **TD-DFT spectra**

   Builds absorption spectra using the Nuclear Ensemble Approximation (NEA)
   from ORCA TD-DFT outputs. It can process individual spectra or folders of
   files, generate convoluted curves, export data, and produce figures or HTML
   reports.

2. **Charge and spin analysis**

   Performs statistical analysis of charge and spin populations from
   trajectories or ensembles. It can work with LIO and ORCA data, including
   Mulliken, Loewdin, Hirshfeld, and CHELPG analyses, histograms, KDE modes,
   time series, and comparisons between systems.

3. **Multilambda kinetics**

   Fits multiwavelength spectrophotometric kinetic experiments. It reads
   absorbance matrices as wavelength vs. time, supports baseline correction,
   wavelength/time selection, spectrum removal, kinetic model fitting, and
   recovery of pure spectra. Available models include `A -> B`, `A -> B -> C`,
   and `A <-> B -> C`, with `nnls`, `pinv`, and `factor` fitting methods.

## Basic Requirements

The main routines are written in Python and mostly rely on:

- Python 3
- NumPy
- SciPy
- Matplotlib

Typical installation:

```bash
python3 -m pip install numpy scipy matplotlib
```

To print the dependency guide included in `TolkienTools`:

```bash
tolkien-tools requirements
```

## Disclaimer

This is an open, public repository, and the code can be reused or adapted.
However, most routines were developed for personal research workflows and may
make assumptions about directory structure, file names, cluster environment,
installed programs, or output formats.

The scripts in `personal_scripts/` should be treated as a personal toolbox
rather than a polished software package. `TolkienTools` is more general and
organized, but it is still research code and may fail on unexpected inputs.
Please inspect the scripts, test them on your own data, and use them at your own
discretion.
