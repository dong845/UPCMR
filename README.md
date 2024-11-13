# UPCMR

Pytorch implementation of the paper "UPCMR: A Universal Prompt-guided Model for Random Sampling Cardiac MRI Reconstruction".

## Data

Download the CMRxRecon2024 dataset, then read each mat file and store the content into the corresponding h5 file, then generate the CSM using EspiritCalib function from sigpy package. "CSM_GU" and "CSM_RA" are from ACS area, while "CSM" is generated from fully sampled case.

## Training and Testing
Before running, some folder pathes need to be specified.
```
python train.py
```
