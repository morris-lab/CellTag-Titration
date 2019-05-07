# CellTag-Titration
Imaging-based titration software for GFP-Celltag lentivirus

### Description
This repository contains a python scripts required for titering for GFP-CellTag Virus.
For a detailed explanation, please refer to the paper: Guo et. al., CellTag Indexing: a genetic barcode-based multiplexing tool for single-cell technologies, bioRxiv, 2018 (new version under review).

For the CellTag experiment, titration of CellTag-lentivirus is required.
Accurate assessment of viral titer can be performed via flow cytometry, we also provide an easy titering method based on the quantification of GFP signal with cell images acquired with a fluorescent microscope. Here, GFP positive cell ratio and CellTag virus titer can be easily calculated by quantifying trypsinized cell images.

The script needs python >=3.6.0 and python libraries listed below.
lease install them before running script in this repository.
We recommend to install them using anaconda. 
 
numpy >= 1.15.4 
scipy >= 1.1.0  
pandas >= 0.23.4
scikit-image >= 0.13.0 
matplotlib >=2.2.3
seaborn >= 0.9.0


### Experimental Method
#### CellTag transduction 
For the information about the detail of CellTag-lentivirus transduction, please refer to the paper; Guo et. al., CellTag Indexing: a genetic barcode-based multiplexing tool for single-cell technologies, bioRxiv, 2018 (new version under review).

#### Image qcquisition
Images are acquired after trypsin treatment. 
Treat cells with trypsin or a reagent for cell dissociation. Please make sure that all cells are completely dissociated.
Load sample into a hemocytometer. Instead of using hemocytometer, cell suspension can be placed on a cell dish.
Take pictures of the cells. This script recquires two kinds 2 types of images: an image in acquired in the bright field and an image of GFP fluorescent signal. Please see the sample images in - https://github.com/morris-lab/CellTag-Titer/test_data.
Please use 4x~10x objective lens. The magnification should be small to acquire a large number of cells per image.


### Image processing
#### Before we start
Clone the Github repository
```
git clone https://github.com/morris-lab/CellTag-Titration.git
cd CellTag-Titration
```
Open jupyter notebook. 
```
jupyter notebook
```
The repository includes sample imaging dataset. 
Please run the jupyter notebook with this dataset first:  -  https://github.com/morris-lab/CellTag-Titer/Quantify_images_of_trypsinized_cells_to_estimate_GFP_positive_ratio.ipynb.



