Primitives
==========================================

This section gives a brief idea about all the primitives used for **SCALES-DRP** pipeline.

.. contents::
   :local:
   :depth: 2

SubtractOverscan
----------------

   **Determines overscan offset and subtracts it from the image.**

   Uses the BIASSEC header keyword to determine where to calculate the overscan
   offset.  Subtracts the overscan offset and records the value in the header.
   In addition, performs a polynomial fit and uses the residuals to determine
   the read noise in the overscan.  Records the overscan readnoise in the
   header as well.


TrimOverscan
------------

   **Trim overscan region from image.**

   Uses the data section (DSECn header keyword) to determine how to trim the
   image to exclude the overscan region.

   Removes raw section keywords, ASECn, BSECn, CSECn, and DSECn after trimming
   and replaces them with the ATSECn keyword giving the image section in the
   trimmed image for each amplifier.

   If the input image is a bias frames, writes out a \*_intb.fits file,
   otherwise, just updates the image in the returned arguments.


MakeMasterBias
--------------

   **Stack bias frames into a master bias frame.**

   Generate a master bias image from overscan-subtracted and trimmed bias
   frames (\*_intb.fits) based on the instrument config parameter
   bias_min_nframes, which defaults to 7.  The combine method for biases is
   'average' and so cosmic rays may be present, especially in RED channel data.
   A high sigma clipping of 2.0 is used to help with the CRs. 
   Uses the ccdproc.combine routine to perform the stacking.

   Writes out a \*_mbias.fits file and records a master bias frame in the proc table.


FlagSaturation
--------------

   **Flag saturated pixels.**

   Currently flags pixels with values > 60,000 counts with a value of 8
   in the flags FITS extension and updates the following FITS header keywords:

      * SATFLAG: set to ``True`` if operation is performed.
      * NSATFLAG: set to the count of saturated pixels.

   Updates the flag extension of the image in the returned arguments.


BadPixel
--------
   
   Working on


FlatFielding
------------
   
   Working on


SubtractBias
------------

   **Subtract the master bias frame.**

   Reads in the master bias created by ``MakeMasterBias`` and performs the
   subtraction (after verifying amplifier configuration agreement).  Records
   the processing in the header.



CorrectGain
-----------

   **Convert raw data numbers to electrons.**

   Uses the ATSECn FITS header keywords to divide image into amp regions and
   then corrects each region with the corresponding GAINn keyword.  Updates the
   following FITS header keywords:

      * GAINCOR: sets to ``True`` if operation performed.
      * BUNIT: sets to `electron`.
      * HISTORY: records the operation.


OptimalExtract
--------------

   **Generate a 3D datacube with wavelength as third axis.**

   Read the bias and flat corrected raw data and perform an 'optimal' extract
   to generate the datacube. 

   User can choose to perform ``optimalExtract`` or ``LeastExtract`` based on the requirments. 





LeastExtract
------------

   **Generate a 3D datacube with wavelength as third axis.**

   Read the bias and flat corrected raw data and perform a 'chi-square' extract
   to generate the datacube with the help of a rectification matrix created using the
   daytime wavelength calibration. 

   User can choose to perform ``optimalExtract`` or  ``LeastExtract`` based on the requirments. 


SkySubtraction
--------------
Working on










