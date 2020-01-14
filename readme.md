# MODIS / Pseudo-MODIS Unmixing

### Authors: Nicholas Wright and Chris Polashenski
#### Last Modified: January 14, 2019

## Table of Contents

* wvunmix.py
* create_sim_tds.py
* create_rfc_model.py
* modis_proc.py

### wvunmix.py

#### Summary

This script reads a raw WV image and its classified (with OSSP) counterpart and performs the pseudo-MODIS unmixing steps. Writes output data to a csv file in the given directory.
    
#### Usage

### create_sim_tds.py / create_rfc_model.py

#### Summary

create_sim_tds.py creates the simulated training set based on the values in albedo.csv. This can be run directly: >>python create_sim_tds.py. This will save a hdf file to the folder where the script exists.

create_rfc_model.py is executed second, and creates the random forest model based on the simulated training set created with create_sim_tds.py. By default it will read the hdf file created earlier.

### modis_proc.py

#### Summary

This script will use the given model (either the random forest created with create_rfc_model or spectral unmixing) to analyze a MODIS image. It will download MODIS data automatically if lib/modis_dl_reproject is modified to include your USGS login information. Otherwise the script can be altered to point to the correct MODIS imagery (each band in a seperate file, naming convention as seen in this script). Enter username/password on lines 405/406 of lib/modis_dl_reproj.py to download automatically. 
