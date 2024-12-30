SCALES-DRP: the SCALES data reduction pipeline
=======================================

Install
.......

git clone https://github.com/scalessim/SCALES-DRP.git

cd SCALES-DRP

pip install -e .

Quick start
...........

Next, go to the data directory and run the startup script:

cd mydata

reduce_kcwi -lr -l target.txt

Where the target.txt has all files to be reduced. Bias and object files. 

The output will be saved in a new directiry called 'redux'





