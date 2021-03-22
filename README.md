Data-processing pipeline to analyze chromatin magnetic manipulation images
==

## About
This GitHub repository contains the code used to process the images acquired using Micro-Manager before subsequent manual analysis.
This version of the pipelin takes as input a MicroManager folder and outputs concatenated movies (one per position).

## Data

- Raw (input) data: raw data is available on Zenodo (upon publication): **URL**
- Concatenated (output) data: 
   - a series of TIFF, 16-bit files, with appropriate (ImageJ-readable) pixel sizes and timestamp overlays
   - a series of .xls, machine-readable files, with timestamp information.
   - available on Zenodo (upon publication): **URL**

## Requirements
1. Use Python 3. 
2. A working [Fiji](https://fiji.sc) installation.
3. Use `pip` to install the missing libraries. We recommend running these scripts in a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/).

```
pip install numpy pandas scipy matplotlib scikit-image
```

## Configuration files & how-to
`.cfg` files are present in the `config/` folder. They contain the parameters used by the script to concatenate the data. It has to be used together with the *raw* data mentioned above, located in the Zenodo repository.

### config.cfg
This file contains most of the user-tunable parameters required to run the pipeline.
- `prefix_dict`: the root of the folder architecture. Within this folder, the raw MicroManager folders should be located in a `data/` subfolder, the concatenated files will be created in the `data/Maxime/concatenation/` subfolder. Make sure that these folders exist. The script can be shared between computers, one entry of the dictionary corresponds to one machine.
- `fiji_dict`: the path to the Fiji executable

## Authors
Maxime Woringer, with inputs from multiple authors & collaborators, and limited & credited Stack Overflow copy-pasting.

## License
GPLv3+, see the LICENSE file included in this repository for more details.
