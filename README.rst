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

The output will be saved in a new directiry called 'redux'





