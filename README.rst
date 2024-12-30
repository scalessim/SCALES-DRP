SCALES-DRP: the SCALES data reduction pipeline
=======================================

Install
.......

The following will create a conda environment called scalesdrp, and activate it.

    conda create --name scalesdrp python=3.7

    conda activate scalesdrp


git clone https://github.com/scalessim/SCALES-DRP.git

cd SCALES-DRP

pip install -e .


Quick start
...........

The assumption is that you have a directory containing SCALES data, and that the names of the files are those assigned at the telescope, lr*.fits and mr*.fits.

Next, go to the data directory and run the startup script:

Give a quick look at the configuration parameters for the pipeline, contained in the file SCALES-DRP/KCWI_DRP-master/kcwidrp/config/kcwi.cfg

For the time being nothing to change.


...........................................................


Next, go to the data directory (mydata) and run the startup script:

cd mydata

reduce_kcwi -lr -l target.txt

Where the target.txt has all files to be reduced. Bias and object files. 


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
twiflat_min_nframes = 1
dark_min_nframes = 3
arc_min_nframes = 1        # = 3 for [MR]
contbars_min_nframes = 1   # = 3 for [MR]
minoscanpix = 75           # = 20 for [MR]
oscanbuf = 20              # = 5 for [MR]


These parameters control the minimum number of bias, internal/dome/twilight flats and darks that the DRP expects before producing a master calibration. 
This minimum numbers are different for the MR and LR channels.


Running the pipeline
....................








