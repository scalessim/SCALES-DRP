SCALES-DRP: the SCALES data reduction pipeline
=======================================

Install
.......

The following will create a conda environment called scalesdrp, and activate it.

    conda create --name scalesdrp python=3.12

    conda activate scalesdrp


git clone https://github.com/scalessim/SCALES-DRP.git


=======
cd SCALES-DRP

pip install -e.


Quick start
...........

The assumption is that you have a directory containing SCALES data, and that the names of the files are those assigned at the telescope, lr*.fits and mr*.fits.

Next, go to the data directory and run the startup script:

Take a quick look at the configuration parameters for the pipeline, contained in the file SCALES-DRP/scalesdrp/config/scales.cfg

For this first run there's no need to change anything


...........................................................


Next, go to the data directory (mydata) and run the startup script:

cd data_scales_drp

start_scales_reduce -lr -l target.txt     # science grade pipeline
start_scales_quicklook -lr -l target.txt  # quicklook pipeline

Where the target.txt has all files to be reduced. Bias and object files.

The daytime calibration will takes place automatically finding the files from a folder. '-d' for directory, followed by the path to the directory, then mention the mode of obseravation. 


start_scales_calib -d path-to-folder -lr  # For daytime calibration process


Three directories will be created: a redux directory with the results of the reduction, a logs directory with separate logs for the framework itself and for the DRP, and a plots directory containing diagnostic plots. Currently nothing is added to 'logs' and 'plots'.


Configuration Parameters
.......................

A number of reduction parameters can be changed using entries in the configuration file.


Low and Medium sections of the configuration file
.................................................
Now that the Low-Resolution (LR) channel has been installed, there is a need to specify different default parameters for each channel. 
These are delineated in the config file with [LR] and [MR] section headers. 

Processing parameters

bias_min_nframes = 7
flat_min_nframes = 6
dome_min_nframes = 3
dark_min_nframes = 3
arc_min_nframes = 3        



These parameters control the minimum number of bias, internal/dome/twilight flats and darks that the DRP expects before producing a master calibration. 
This minimum numbers are different for the MR and LR channels.


Running the pipeline
....................

User must specify which channel to process with the -lr for lowres mode or -mr for medres mode.

reduce_scales -lr -f lr_file.fits

To reduce a lowres file. Here -lr for lowres and -f for file followed by the filename.

reduce_scales -lr -l lr_file.txt

To reduce a list of files together. Here -lr for lowres mode, -l for list followed by the text files with the names of the fits file to reduce. 









