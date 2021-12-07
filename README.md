# Multi-object Filters demo

This is a README file for the Python implementation of a set of multiple multi-object filters based
on point processes (or random finite sets), using Gaussian mixtures, along with a demonstration script to compare the filters. In particular, the following filters have been implemented:
  - Probability Hypothesis Density (PHD) filter
  - Cardinalized PHD (CPHD) filter
  - Discrete-gamma with Marks (DGM) filter (DG filter with labels)
  - Generalized Labeled Multi-Bernoulli (GLMB) filter
  - Joint GLMB (JGLMB) filter (with joint prediction and update steps)
  - Linear Complexity Cumulant (LCC) filter
  - Linear Complexity Cumulant with Marks (LCCM) filter (LCC filter with labels)

 The implementation of the PHD, CPHD, GLMB and JGLMB filters are based on the Matlab implementation of those filters made public in [Ba Tuong Vo's web page](http://ba-tuong.vo-au.com/codes.html). The implementation of the CPHD filter has been changed by the owner of this repository to make it more efficient by precomputing many factors necessary in the prediction and update of the cardinality distribution.

## Licensing
This code is licensed by Flávio Eler De Melo under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0.html). A copy of the terms and conditions can be found in the [LICENSE](LICENSE) file.

## Contact
Please get in touch by messages for addressing doubts or suggestions. If you have found a bug, please open a ticket.

## Requirements
  - Python >= 3.5
  - Numpy
  - Scipy
  - Termcolor
  - Argparse
  - Munkres >= 1.0.12
  - Pickle5
  - Nuitka
  - PyYAML

## Installation

### **1.** Clone the repository

```
git clone https://github.com/femelo/multi-object-filters
```

### **2.** Install Python dependencies

Change directory to the code base directory
```
cd multi-object-filters
```

and install required packages
```
python3 -m pip install -r requirements.txt
```

## Usage

For example, to run the PHD, CPHD and LCC filters (and compare their outputs) with the default parameters in verbose mode, type
```
python3 demo.py -f phd cphd lcc -v
```

All options for the demo script can be seen by checking the help as:
```
python3 demo.py -h
```

All figures for performance evaluation are saved in the folder 'figures/'.

## Precompilation

The application can be precompiled to run faster via Nuitka. First install the *patchelf* package:
```
sudo apt-get install patchelf
```

Then, for the compilation, just run the bash script:
```
./build.sh
```

The created standalone application *demo.bin* can be called as, for instance,
```
./demo.bin -f phd cphd lcc -v
```
