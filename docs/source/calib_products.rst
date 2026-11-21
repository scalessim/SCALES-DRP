
.. _calibproducts:

Calibration Products
====================

**Generates the master calibration and detector reference files
required for the SCALES Data Reduction Pipeline (SCALES-DRP).**

The SCALES-DRP uses a set of calibration reference files to
correct detector effects, characterize instrumental response,
and perform spectral extraction. These calibration products
are generated from dedicated calibration observations and
are subsequently used during science-grade data reduction.

The calibration products are generated through two
complementary calibration modules.

* :ref:`Calibration Module-I <calibmodule1>` processes
  calibration exposures to generate master calibration
  products, including master dark frames, master bias
  frames, master detector flats, and processed
  monochromator exposures at individual wavelengths.

* :ref:`Calibration Module-II <calibmodule2>` generates
  additional detector and spectral calibration products,
  including read-noise maps, saturation maps, linearity
  coefficients, bad-pixel maps, rectification matrices,
  and lenslet flats.

These calibration products are used at different stages
of the reduction process. Detector calibration files
support reference-pixel correction, linearity correction,
ramp fitting, bad-pixel correction, and flat-fielding.
Spectral calibration products provide the information
required to reconstruct the spatial and spectral
distribution of the incident light and extract
three-dimensional IFS data cubes.

The calibration procedures are implemented as individual
primitives, allowing the corresponding reference files
to be generated and updated independently when new
calibration observations become available.

The following sections describe the calibration products,
their generation procedures, and the information stored
in the resulting reference files. The calibration procedures
can be customized using the :ref:`configuration parameters <configparams>`.
Each primitive also generates associated diagnostic plots
to facilitate verification and assessment of the calibration results.


The calibration primitives are located in:

.. code-block:: text

   SCALES-DRP/scalesdrp/Primitives/

The generated calibration reference files are stored
in the configured calibration directory, typically:

.. code-block:: text

   SCALES-DRP/scalesdrp/calib/

.. contents::
   :local:
   :depth: 2



.. _createreadnoise:

Read-Noise Map Generation
-------------------------

**Estimates the per-pixel single-read detector noise from a set of
pre-corrected dark exposures using correlated double sampling (CDS).**

This primitive generates a two-dimensional read-noise map for the
SCALES detectors. The input is a stack of dark ramp cubes that has
already undergone detector reference-pixel, alternating-column-noise
(ACN), and 1/f-noise corrections. The routine constructs CDS images
from two selected reads of each dark exposure and estimates the
pixel-to-pixel variation in temporal noise across the exposures.

The implementation provides two noise-estimation methods:
``pair_diff``, based on differences between independent CDS
exposures, and ``stack_std``, based on the temporal standard
deviation of the CDS stack. Both methods return the noise of a
**single detector read** in the unit of data numbers (DN).


Calibration Procedure
^^^^^^^^^^^^^^^^^^^^^

The read-noise calibration follows these steps:

1. **Input validation:** The routine accepts a four-dimensional
   stack with shape ``(n_exposures, n_reads, ny, nx)`` and validates
   the selected read indices, estimation method, gain, clipping
   threshold, and minimum sample count. The ``pair_diff`` method
   requires at least four exposures and enough exposures to form
   ``min_good`` independent pairs.

2. **CDS construction:** For each exposure, a CDS image is formed
   by subtracting the selected earlier read from the later read.
   Only the two selected reads are converted to floating-point
   arrays, avoiding an unnecessary copy of the entire ramp stack.

3. **Residual offset subtraction:** Each CDS image can undergo
   additional median-based subtraction of a global offset,
   amplifier-dependent offsets, and row-dependent offsets.
   The medians are estimated from finite active detector pixels,
   excluding the configured reference-pixel border.

4. **Noise-sample construction:** The routine either forms
   non-overlapping differences between independent CDS exposures
   (``pair_diff``) or uses the individual CDS exposures directly
   (``stack_std``).

5. **Robust noise estimation:** For each detector pixel, the
   temporal sample distribution is sigma-clipped using a median
   and median absolute deviation (MAD). The sample standard
   deviation is then calculated from the retained values.

6. **Single-read noise conversion:** The measured standard
   deviation is divided by the appropriate factor to convert
   from CDS or CDS-pair differences to single-read noise.

7. **Unreliable-pixel identification:** Pixels with too few
   retained samples, non-finite noise, or nonpositive noise are
   flagged. The reference-pixel border can also be excluded
   explicitly when requested.

8. **Spatial interpolation:** Unreliable read-noise values are
   replaced using neighboring reliable pixels. The corresponding
   bad-read-noise mask retains the locations of all replaced
   or excluded pixels.

9. **Final validation:** The routine checks that the interpolated
   single-read noise map and derived CDS-noise map contain finite,
   physically positive values. It also prepares a FITS header
   describing the calibration inputs and processing settings.

CDS Noise Estimation
^^^^^^^^^^^^^^^^^^^^

For exposure :math:`j`, the CDS image is constructed as:

.. math::

  C_j(x,y) = D_{j,r_1}(x,y) - D_{j,r_0}(x,y),

where :math:`D_{j,r_0}` and :math:`D_{j,r_1}` are the selected
reads of the dark ramp, with :math:`r_1 > r_0`.

Assuming the two reads have independent, equal-variance read noise,
and that dark-current shot noise is negligible, the CDS variance is:

.. math::

  \operatorname{Var}(C_j) = 2\sigma_{\mathrm{read}}^2.

Two methods are available to estimate the single-read noise.

**Independent CDS-pair differences (``pair_diff``)**

The CDS exposures are grouped into non-overlapping pairs:
``(0, 1)``, ``(2, 3)``, ``(4, 5)``, and so on. For each pair,
the routine constructs:

.. math::

  P_k(x,y) = C_{2k+1}(x,y) - C_{2k}(x,y).

If the CDS exposures are independent, then:

.. math::

  \operatorname{Var}(P_k) = 4\sigma_{\mathrm{read}}^2.

The single-read noise is therefore:

.. math::

  \sigma_{\mathrm{read}}(x,y)
  = \frac{\operatorname{std}_k[P_k(x,y)]}{2}.

The implementation uses the sample standard deviation after
sigma clipping. An unpaired final exposure is not used when
the number of exposures is odd.

**CDS-stack standard deviation (``stack_std``)**

Alternatively, the temporal standard deviation of the CDS
exposures is calculated directly:

.. math::

  \sigma_{\mathrm{read}}(x,y)
  = \frac{\operatorname{std}_j[C_j(x,y)]}{\sqrt{2}}.

This method uses the full CDS stack rather than forming
independent exposure pairs.

In both cases, the equivalent CDS noise is reconstructed as:

.. math::

  \sigma_{\mathrm{CDS}}(x,y)
  = \sqrt{2}\,\sigma_{\mathrm{read}}(x,y).


Robust Sample Selection
^^^^^^^^^^^^^^^^^^^^^^^

The routine estimates a robust central value and scatter for
each pixel across the noise-sample stack. The initial scatter
estimate is based on the median absolute deviation:

.. math::

  \sigma_{\mathrm{MAD}}
  = 1.4826\,\operatorname{median}
  \left(\left|X_k-\operatorname{median}(X_k)\right|\right),

where :math:`X_k` denotes the CDS or CDS-pair difference
samples for a particular pixel.

If the MAD estimate is zero or invalid, the sample standard
deviation is used as a fallback scale. Finite samples are
retained when their deviations from the median are within
the configured clipping threshold:

.. math::

  |X_k-\operatorname{median}(X_k)|
  \leq n_\sigma\,\sigma_{\mathrm{scale}}.

The standard deviation is computed using the retained samples
with one degree of freedom removed (``ddof=1``). The number
of retained samples is recorded separately for every pixel.
Pixels with fewer than ``min_good`` samples are considered
unreliable.

Unreliable Read-Noise Pixels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The routine identifies pixels for which a reliable read-noise
estimate cannot be obtained. These include pixels with:

* Fewer than the required number of valid samples.
* Non-finite or nonpositive read-noise estimates.
* Reference-pixel locations, when ``exclude_ref_pixels`` is enabled.

The resulting Boolean mask, ``bad_rn``, identifies all unreliable
or excluded pixels. Invalid values are first seeded using the
nearest originally reliable pixel and then replaced using a
local median-filtered value. The interpolation uses a configurable
odd-sized spatial kernel, with a default width of three pixels.

The final map is required to contain finite, strictly positive
values. Interpolated pixels remain flagged in ``bad_rn`` so that
measured and substituted noise estimates can be distinguished.

Calibration Products
^^^^^^^^^^^^^^^^^^^^

The routine returns the following calibration arrays and metadata:

.. list-table:: Read-noise calibration products
  :header-rows: 1
  :widths: 25 75

  * - Product
    - Description
  * - ``rn_e``
    - Two-dimensional single-read noise map. With the pipeline
      gain set to unity, values are interpreted in DN.
  * - ``cds_noise_dn``
    - Equivalent two-dimensional CDS noise map in DN.
  * - ``n_good``
    - Number of retained noise-estimation samples per pixel
      after sigma clipping.
  * - ``bad_rn``
    - Boolean map identifying unreliable, excluded, or
      interpolated read-noise pixels.
  * - ``hdr``
    - FITS header containing the calibration method, read
      indices, number of exposures, clipping settings,
      amplifier/reference-pixel settings, and interpolation
      provenance.

The header includes calibration keywords such as ``CALTYPE``,
``RNMETHOD``, ``NFILES``, ``NPAIRS``, ``READ0``, ``READ1``,
``CLIPSIG``, ``MINGOOD``, ``NCHANS``, ``REFWID``, ``RNINTERP``,
and ``NBADRN``.


Output
^^^^^^

The primitive computes a per-pixel single-read noise map,
an equivalent CDS-noise map, a valid-sample count map,
an unreliable-pixel mask, and a calibration FITS header.
These outputs can be written into a master read-noise
reference file by the calling calibration pipeline.

The related code can be found at:

.. code-block:: text

  SCALES-DRP/scalesdrp/Primitives/readnoise_create.py



.. _createsat:

Creating Saturation Map
-----------------------

**Generates a per-pixel detector saturation-threshold map from multiple
flat-field ramps using sustained decreases in consecutive-read differences.**

The SCALES-DRP estimates the saturation level of each detector pixel from
flat-field exposures containing multiple nondestructive reads. As a pixel
approaches saturation, its accumulated signal no longer increases at the
rate measured during the early part of the ramp. The algorithm identifies
the first sustained decrease in the read-to-read signal increment and
records the corresponding measured signal as the saturation threshold.

The input ramps have shape ``(n_ramps, n_reads, nrows, nx)`` and are
expected to have undergone the appropriate ACN and 1/f corrections.
All signal values and saturation thresholds are expressed in data
numbers (DN). The resulting saturation map uses **absolute detector DN**;
no reset-pedestal subtraction is performed during its creation.


Calibration Procedure
^^^^^^^^^^^^^^^^^^^^^

The saturation-map generation follows these steps:

1. **Input validation:** The routine checks the dimensions of the
   calibration ramp stack and the requested detection parameters.
   Each ramp must contain at least ``min_sat_reads``.

2. **Read-difference calculation:** Consecutive-read differences are
   computed independently for each ramp and detector pixel.

3. **Early-ramp reference estimation:** A median reference increment is
   calculated from a configurable interval of early differences.
   The reference is accepted only if enough finite samples are
   available and its value exceeds a minimum signal threshold.

4. **Sustained-drop detection:** Each difference is compared with a
   configurable fraction of the early reference increment. The first
   sequence of sufficiently many consecutive differences below this
   threshold marks the saturation onset for that ramp and pixel.

5. **Saturation measurement:** The absolute DN value of the first
   affected read is recorded as the saturation estimate for the ramp.
   A corresponding zero-based read index is also recorded.

6. **Combination across ramps:** Valid saturation measurements are
   combined using their median to produce a two-dimensional raw
   saturation map. Detection counts, detection fractions, and
   ramp-to-ramp scatter are calculated independently.

7. **Quality assessment:** Pixels are flagged when measurements are
   unavailable, implausible, detected too early, inconsistent between
   ramps, or supported by too few observations.

8. **Quality-aware interpolation:** Rejected or missing thresholds are
   replaced using the medians of valid neighboring pixels. An optional
   final fallback can fill regions that remain unresolved.

9. **Reference-file generation:** The final map, original measurements,
   diagnostic maps, and data-quality definitions can be written to a
   multi-extension FITS file.


Saturation Detection
^^^^^^^^^^^^^^^^^^^^

For each pixel in ramp :math:`r`, the read-to-read difference is

.. math::

   \Delta D_{r,i} = D_{r,i+1} - D_{r,i},

where :math:`D_{r,i}` is the measured accumulated signal in DN at read
:math:`i`. Read indices are zero-based.

The normal early-ramp increment is estimated as

.. math::

   \Delta D_{\mathrm{ref},r}
   = \operatorname{median}_{i\in\mathcal{R}}
   \left(\Delta D_{r,i}\right),

where :math:`\mathcal{R}` is the selected interval of finite early
read differences. By default, the algorithm starts at difference index
1, considers up to 10 differences, and requires at least 3 finite
values. A reference is usable only if

.. math::

   \Delta D_{\mathrm{ref},r}
   \geq \Delta D_{\mathrm{min}},

where the default minimum reference increment is 1 DN.

The saturation search identifies differences satisfying

.. math::

   \Delta D_{r,i}
   < f_{\mathrm{drop}}\,\Delta D_{\mathrm{ref},r},

where :math:`f_{\mathrm{drop}}` is the fractional threshold, with a
default of ``drop_fraction``. The condition must remain true for a configurable
number of consecutive differences.

If the first qualifying difference occurs at index :math:`k`, the
first affected read is :math:`k+1`. The saturation measurement for
that ramp is therefore

.. math::

   S_{r} = D_{r,k+1},

with saturation-read index

.. math::

   i_{\mathrm{sat},r} = k+1.

By default, the search includes the entire ramp, including the early
reference interval. The optional ``search_after_reference=True``
setting restricts the search to differences after that interval.
Detections overlapping the reference interval or occurring before the
configured initial-read cutoff are flagged as early saturation.

.. note::

   The sustained-drop criterion compares consecutive-read differences,
   but the stored saturation threshold is the **absolute DN value of
   the first affected read**. It is not a difference value and is not
   pedestal-subtracted. A sustained drop can be detected before a
   completely flat saturation plateau develops.

Combining Measurements Across Ramps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The raw per-pixel saturation threshold is the median of finite
saturation measurements from the calibration ramps:

.. math::

   S_{\mathrm{raw}}(x,y)
   = \operatorname{median}_{r\in\mathcal{D}(x,y)}
   S_r(x,y),

where :math:`\mathcal{D}(x,y)` contains the ramps with finite
saturation detections for that pixel.

The algorithm also records the number of ramps with a usable early
reference, :math:`N_{\mathrm{valid}}`, and the number with finite
saturation detections, :math:`N_{\mathrm{detected}}`. Their ratio is

.. math::

   f_{\mathrm{det}}
   = \frac{N_{\mathrm{detected}}}{N_{\mathrm{valid}}}.

The detection fraction is undefined when no ramp has a usable
reference. By default, a detection fraction below 0.5 or fewer than
three detected ramps triggers a ``LOW_CONFIDENCE`` flag.

Ramp-to-ramp scatter is estimated robustly as

.. math::

   \sigma_{\mathrm{sat}}
   = 1.4826\,\operatorname{median}_{r}
   \left|S_r-S_{\mathrm{raw}}\right|.

The scatter is reported only when at least three ramps have finite
saturation detections. A pixel is flagged as inconsistent if

.. math::

   \frac{\sigma_{\mathrm{sat}}}{|S_{\mathrm{raw}}|}
   > f_{\mathrm{scatter,max}},

where the default maximum relative scatter is 0.10.

The ``SAT_READ`` map contains the rounded median zero-based read index
of the detected saturation onsets. This median index and the median
saturation DN are computed independently; they need not correspond to
the same individual calibration ramp.

Quality Assessment and Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The routine preserves the original saturation measurements in
``SATURATION_RAW`` and creates a ``SAT_QUALITY`` bitmask for each pixel.
Measured thresholds below the minimum accepted saturation level
or above an optional maximum are flagged. Additional
flags identify early detections, insufficient usable ramps, and
inconsistent measurements.

Before interpolation, the default rejection mask excludes measurements
with ``SATURATION_TOO_LOW``, ``SATURATION_TOO_HIGH``,
``EARLY_SATURATION``, ``INCONSISTENT_RAMPS``, or
``TOO_FEW_VALID_RAMPS``. Low-confidence measurements remain eligible
by default, but can also be rejected by setting
``reject_low_confidence=True``.

The interpolation procedure fills rejected or missing pixels with the
median of their available eight neighboring values. Updates are made
synchronously during each iteration, without wrapping values across
detector edges. The default limit is five iterations.

If pixels remain unresolved, the optional ``fallback_saturation_dn``
parameter can supply a fixed DN value or the median of the originally
accepted measurements. By default, this fallback is ``None``, so
unresolved pixels remain ``NaN``. The routine preserves provenance
masks indicating which pixels were measured, neighbor-filled,
globally filled, or unresolved.

.. note::

   Interpolated values replace unreliable entries only in the final
   ``SATURATION`` map. The original measurements, quality flags,
   detection fractions, and scatter diagnostics are preserved.


Data Quality Flags (DQ Flags)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``SAT_QUALITY`` extension stores a bitmask describing the
reliability of the saturation measurement. The FITS file also contains
a ``DQ_DEFINITIONS`` table with the corresponding flag values.

.. list-table:: Saturation calibration quality flags
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``NONFINITE``
     - Non-finite reference increment or saturation measurement.
   * - ``LOW_REFERENCE_DIFF``
     - Early reference increment is below the minimum usable signal.
   * - ``NO_SATURATION_FOUND``
     - No finite saturation onset was detected.
   * - ``FALLBACK_USED``
     - The final threshold was obtained through interpolation or fallback.
   * - ``SATURATION_TOO_LOW``
     - Measured threshold is below the configured minimum DN.
   * - ``SATURATION_TOO_HIGH``
     - Measured threshold exceeds the optional maximum DN.
   * - ``LOW_CONFIDENCE``
     - Detection fraction or number of detected ramps is too small.
   * - ``TOO_FEW_VALID_RAMPS``
     - Insufficient ramps have usable early reference increments.
   * - ``EARLY_SATURATION``
     - A detected onset is earlier than the initial-read cutoff or
       overlaps the reference interval.
   * - ``INCONSISTENT_RAMPS``
     - Relative robust scatter exceeds the configured limit.
   * - ``NEIGHBOR_FILL_USED``
     - The final threshold was filled using neighboring pixels.
   * - ``GLOBAL_FILL_USED``
     - The final threshold was filled using the optional global fallback.

Calibration Products
^^^^^^^^^^^^^^^^^^^^

The saturation calibration reference file contains the final
per-pixel threshold map, original measurements, quality information,
and diagnostics. The FITS writer uses the following extension names:

.. list-table:: Saturation calibration reference products
   :header-rows: 1
   :widths: 30 70

   * - FITS Extension
     - Description
   * - ``SATURATION``
     - Final per-pixel absolute saturation threshold in DN, after
       quality-aware filling.
   * - ``SATURATION_RAW``
     - Original measured median saturation threshold in DN, before
       rejection or interpolation.
   * - ``SAT_READ``
     - Rounded median zero-based saturation-read index; -1 indicates
       no detection.
   * - ``SAT_QUALITY``
     - Per-pixel quality bitmask, including interpolation flags.
   * - ``REF_DIFF``
     - Median early-ramp reference increment in DN.
   * - ``NDETECTED``
     - Number of ramps with finite saturation detections.
   * - ``NVALIDRAMP``
     - Number of ramps with usable early reference increments.
   * - ``SATSCAT``
     - Robust scatter of measured saturation thresholds in DN.
   * - ``SATFRAC``
     - Fraction of usable ramps with detected saturation.
   * - ``ORIGINVAL``
     - Mask identifying originally invalid or rejected measurements.
   * - ``NEIGHFILL``
     - Mask identifying pixels filled using neighboring values.
   * - ``GLOBALFILL``
     - Mask identifying pixels filled by the optional global fallback.
   * - ``UNRESOLV``
     - Mask identifying pixels that remain non-finite after filling.
   * - ``DQ_DEFINITIONS``
     - Table listing the saturation-quality flag names and bit values.

The primary FITS header records the calibration parameters, input
exposure provenance, and detector metadata when supplied. In
particular, ``SATCONV='ABSOLUTE'`` records the absolute-DN convention.


Output
^^^^^^

The calibration procedure returns a dictionary containing the final
saturation map, raw measurements, quality flags, and diagnostic maps.
The ``create_saturation_map()`` function does not itself write a file;
``write_saturation_fits()`` can be used to save the reference products.


The related code can be found at:

.. code-block:: text

   SCALES-DRP/scalesdrp/Primitives/saturation_create.py



.. _createlinearity:

Linearity Coefficient Generation
--------------------------------

**Generates per-pixel detector nonlinearity calibration coefficients
from uniformly illuminated flat-field ramps.**

This primitive derives the detector nonlinearity coefficients used
by the :ref:`linearity` primitive. The SCALES-DRP implements a
`Brandt-style <https://github.com/RomanSpaceTelescope/SOCReferenceFileCode>`_
polynomial calibration approach, in which the nonlinear response
of each detector pixel is determined from multiple ACN and 1/f corrected flat-field ramps.

The calibration fits a common nonlinear response for each pixel
across the available detector ramps, while allowing each ramp
to have an independent linear slope and intercept. This separates
the detector's nonlinear response from variations in illumination
and initial detector signal between exposures.

The correction polynomial is expressed as:

.. math::

   L(M) = M + \sum_{k \in K} c_k M^k,

where :math:`M` is the pedestal-subtracted measured signal,
:math:`L(M)` is the corresponding linearized signal,
:math:`c_k` are the nonlinear correction coefficients, and
:math:`K` represents the selected polynomial powers.

The constant and linear coefficients are fixed at
:math:`c_0=0` and :math:`c_1=1`, respectively. The higher-order
coefficients are determined from the calibration data.

The polynomial orders are configurable through
``polynomial_terms``. For example, ``(2, 3, 4)`` fits the
quadratic, cubic, and fourth-order terms. Polynomial powers
not included in the selected terms are assigned zero coefficients.

Calibration Procedure
^^^^^^^^^^^^^^^^^^^^^

The coefficient generation follows these steps:

1. **Input validation:** The routine validates the input
   flat-field ramps, saturation map, bad-pixel map,
   and read-noise information. Pixels identified by
   the input bad-pixel map are excluded from calibration.

2. **Pedestal estimation:** A separate pedestal is estimated
   for each pixel and calibration ramp using the median
   of the selected initial detector reads. The pedestal
   is subtracted from the measured ramp before fitting
   the nonlinear response.

3. **Saturation masking:** The usable signal range is
   determined using the per-pixel saturation map. A
   configurable fraction of the pedestal-subtracted
   saturation level defines the maximum signal included
   in the fit. Once a ramp reaches this threshold,
   that read and all subsequent reads are excluded.

4. **Plateau detection:** Consecutive-read differences
   are examined to identify sustained reductions in
   the measured signal increment. Reads beginning
   at a detected saturation plateau, together with
   all subsequent reads, are excluded from fitting.

5. **Read selection:** Initial reads affected by
   detector reset transients are excluded. The routine
   also rejects non-finite samples and optionally
   excludes negative pedestal-subtracted signals.
   Ramps with insufficient valid reads are discarded.

6. **Weighted polynomial fitting:** A common set of
   nonlinear coefficients is fitted for each pixel
   across the available calibration ramps. Independent
   linear slopes and intercepts are fitted for each
   ramp. The fit uses a variance model incorporating
   detector read noise and photon noise.

7. **Iterative outlier rejection:** The routine
   performs optional sigma clipping of the
   variance-normalized fitting residuals. Samples
   exceeding the selected threshold are excluded,
   and the polynomial model is refitted using
   the remaining valid samples.

8. **Physical validation:** The fitted polynomial
   is evaluated over the calibrated signal interval.
   The routine checks the polynomial derivative
   and optionally rejects solutions with excessive
   correction amplitudes, large derivatives, or
   poor reduced chi-square values.

9. **Fallback correction:** Pixels with insufficient
   valid samples, unsuccessful fits, or solutions
   failing the selected quality criteria are
   assigned identity coefficients, corresponding
   to no nonlinear correction.


Saturation and Plateau Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Saturated detector reads are excluded from the polynomial
coefficient fit. The calibration uses a previously generated
two-dimensional :ref:`saturation map <createsat>` containing the saturation
level of each detector pixel in absolute DN.

For each calibration ramp, the pedestal-subtracted
measured signal is:

.. math::

   M_{r,i} = D_{r,i} - P_r,

where :math:`D_{r,i}` is the measured detector signal
in read :math:`i` of ramp :math:`r`, and :math:`P_r`
is the estimated pedestal for that ramp.

The maximum usable pedestal-subtracted signal is:

.. math::

   S_{\mathrm{usable},r}
   =
   f_{\mathrm{sat}}
   (S_{\mathrm{sat}} - P_r),

where :math:`S_{\mathrm{sat}}` is the per-pixel saturation
level in absolute DN and :math:`f_{\mathrm{sat}}` is
the configurable saturation fraction.

The first read satisfying

.. math::

   M_{r,i} \geq S_{\mathrm{usable},r}

is identified as the saturation threshold crossing.
That read and all subsequent reads are excluded
from coefficient fitting.

An additional plateau-detection algorithm examines
the differences between consecutive reads:

.. math::

   \Delta M_{r,i}
   =
   M_{r,i+1} - M_{r,i}.

A reference signal increment is estimated from the
median of the selected early read differences:

.. math::

   \Delta M_{\mathrm{ref}}
   =
   \operatorname{median}
   (\Delta M_{r,i})_{\mathrm{early}}.

The algorithm searches for a sustained reduction
in the signal increment relative to this reference.

For each candidate window of consecutive-read
differences, two conditions are evaluated:

* The absolute mean signal increment within the
  window must fall below a configurable fraction
  of the early reference increment.

* A configurable fraction of the differences
  within the window must have absolute values
  below a second threshold defined relative
  to the reference increment.

The window must also contain a sufficient number
of finite differences. The search begins after
the early reference region.

The first window satisfying these conditions
identifies the onset of a sustained plateau.
The corresponding read and all subsequent
reads are excluded from the polynomial fit.

The saturation-map threshold and plateau detection
provide complementary safeguards against including
saturated or nearly saturated detector samples.


Polynomial Fitting and Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nonlinear coefficients are obtained through
a weighted linear least-squares fit using the valid
samples from multiple detector ramps.

For each pixel, the model is:

.. math::

   M_{r,i}
   +
   \sum_{k \in K} c_k M_{r,i}^{k}
   =
   a_r + b_r t_{r,i},

where :math:`M_{r,i}` is the pedestal-subtracted
measured signal in read :math:`i` of ramp :math:`r`,
:math:`a_r` and :math:`b_r` are the fitted intercept
and slope of that ramp, and :math:`t_{r,i}` is
the corresponding read time.

The nonlinear coefficients :math:`c_k` are shared
across all calibration ramps for a given pixel,
while the linear parameters are fitted separately
for each ramp.

The equation can be rearranged as:

.. math::

   \sum_{k \in K} c_k M_{r,i}^{k}
   - a_r - b_r t_{r,i}
   =
   -M_{r,i}.

This forms a linear system in the unknown polynomial
coefficients, ramp intercepts, and ramp slopes.

The system is solved using weighted least squares
with ``numpy.linalg.lstsq``. The design-matrix
columns are normalized before fitting to improve
numerical conditioning.

The fitting weights are derived from a variance
model combining detector read noise and photon noise:

The variance model is:

.. math::

   \sigma^2(M)
   =
   \sigma_{\mathrm{read}}^2
   +
   \max(M,0),

where :math:`M` is the pedestal-subtracted measured
signal in DN and :math:`\sigma_{\mathrm{read}}`
is the detector read noise in DN.

The weighted chi-square statistic is:

.. math::

   \chi^2
   =
   \sum_{r,i}
   \frac{
   \left[
   L(M_{r,i}) -
   (a_r+b_r t_{r,i})
   \right]^2
   }{\sigma_{r,i}^2}.

The reduced chi-square is calculated as:

.. math::

   \chi_{\nu}^2
   =
   \frac{\chi^2}{N_{\mathrm{valid}}-N_{\mathrm{par}}},

where :math:`N_{\mathrm{valid}}` is the number of
valid fitting samples and :math:`N_{\mathrm{par}}`
is the total number of fitted parameters.

The number of parameters is:

.. math::

   N_{\mathrm{par}}
   =
   N_{\mathrm{terms}} + 2N_{\mathrm{ramps}},

where :math:`N_{\mathrm{terms}}` is the number
of selected nonlinear polynomial powers and
:math:`N_{\mathrm{ramps}}` is the number of
calibration ramps included in the model.

The fitting procedure supports iterative sigma
clipping. Following each fit, residuals are
normalized by their corresponding standard
deviations. Samples exceeding the selected
sigma threshold are removed, and the fit
is repeated.

The final coefficients are evaluated for physical
consistency. In particular, the derivative of
the correction polynomial is:

.. math::

   \frac{dL}{dM}
   =
   1 +
   \sum_{k \in K} k c_k M^{k-1}.

The derivative is evaluated over the calibrated
signal interval to verify that the correction
remains monotonically increasing within the
selected tolerance.

Additional optional quality criteria can reject
solutions with excessive reduced chi-square,
correction amplitude, or polynomial derivative.

If the calibration fails the selected criteria,
the pixel is assigned identity coefficients:

.. math::

   L(M) = M.

The corresponding quality flags are recorded
in the calibration reference file.

Calibration Products
^^^^^^^^^^^^^^^^^^^^

The generated linearity calibration FITS file
contains the polynomial coefficients, saturation
information, fitting diagnostics, and quality maps.

.. list-table:: Linearity calibration reference products
   :header-rows: 1
   :widths: 25 75

   * - FITS Extension
     - Description
   * - ``COEFF``
     - Per-pixel polynomial coefficients stored in
       ascending order of polynomial power.
   * - ``SATURATION``
     - Per-pixel saturation level in absolute DN.
   * - ``BPM``
     - Input bad-pixel mask used during calibration.
   * - ``QUALITY``
     - Per-pixel bitmask describing calibration
       failures and quality conditions.
   * - ``SUCCESS``
     - Boolean map identifying pixels with
       successful coefficient fits.
   * - ``CHISQ``
     - Chi-square statistic of the polynomial fit.
   * - ``REDCHISQ``
     - Reduced chi-square statistic.
   * - ``NVALID``
     - Number of valid samples used in fitting.
   * - ``DOF``
     - Degrees of freedom of the fit.
   * - ``PEDESTAL``
     - Estimated pedestal for each calibration
       ramp and detector pixel.
   * - ``SLOPE``
     - Fitted linear slope for each calibration
       ramp and detector pixel.
   * - ``INTERCEPT``
     - Fitted linear intercept for each
       calibration ramp and detector pixel.
   * - ``RESIDRMS``
     - Root-mean-square residual of the fit.
   * - ``MINDERIV``
     - Minimum polynomial derivative over
       the tested signal interval.
   * - ``NITER``
     - Number of fitting iterations completed.
   * - ``READNOISE``
     - Detector read-noise map used for calibration.
   * - ``DQDEF``
     - Table defining the quality-flag names
       and corresponding bit values.

The ``READNOISE`` extensions is
included when the corresponding maps are supplied
to the FITS-writing routine.

The ``COEFF`` and ``SATURATION`` extensions are
used by the :ref:`linearity` primitive when
correcting detector ramps.

The additional extensions provide diagnostic
information for evaluating calibration quality
and identifying pixels with unreliable
nonlinearity corrections.

Data Quality Flags (DQ Flags)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Records the quality and validity of the per-pixel
linearity calibration.**

During coefficient generation, the DRP creates
a per-pixel quality map stored in the ``QUALITY``
extension of the calibration FITS file.

Each pixel can be assigned one or more bitmask
flags describing the outcome of the calibration.
The definitions and corresponding bit values
are recorded in the ``DQDEF`` extension.

The quality flags include:

* **Input and calibration validity**

  - ``INPUT_BPM``: Pixel excluded by the input
    bad-pixel mask.
  - ``NONFINITE_INPUT``: Non-finite values detected
    in the input calibration ramps.
  - ``INVALID_SATURATION``: Invalid or unavailable
    saturation level.
  - ``INVALID_READNOISE``: Invalid read-noise value.
  - ``TOO_FEW_SAMPLES``: Insufficient valid samples
    or calibration ramps for fitting.

* **Polynomial fitting**

  - ``FIT_FAILED``: Numerical fitting failure,
    rank-deficient system, or invalid fitting solution.
  - ``NONFINITE_COEFFICIENT``: Invalid polynomial
    coefficient or derivative evaluation.
  - ``HIGH_CHISQ``: Reduced chi-square exceeds
    the selected quality threshold.

* **Physical validation**

  - ``NONMONOTONIC``: Fitted correction fails
    the monotonicity criterion.
  - ``EXCESSIVE_CORRECTION``: Correction amplitude
    exceeds the permitted fraction.
  - ``EXCESSIVE_DERIVATIVE``: Polynomial derivative
    exceeds the selected maximum value.

* **Fallback calibration**

  - ``IDENTITY_FALLBACK``: The pixel is assigned
    identity coefficients because a reliable
    nonlinear correction could not be obtained.

The quality flags can be combined for individual
pixels, allowing multiple calibration conditions
to be recorded simultaneously.

In addition to the ``QUALITY`` map, the calibration
file contains a ``SUCCESS`` extension identifying
pixels for which the polynomial fit passed the
selected validation criteria.

Pixels assigned identity coefficients retain
their original measured signals when the
linearity correction is subsequently applied.

Output
^^^^^^

The primitive produces a multi-extension FITS
linearity reference file containing the per-pixel
polynomial coefficients, saturation map, quality
flags, and associated fitting diagnostics.

The coefficient planes are stored in ascending
polynomial order, with the constant and linear
coefficients fixed at zero and unity, respectively.

The calibration reference file is subsequently
used by the :ref:`linearity` primitive to
linearize detector ramps before ramp fitting.

The related code can be found at:

.. code-block:: text

   SCALES-DRP/scalesdrp/Primitives/linearity_create.py


.. _createbpm:

Creating Bad Pixel Mask
-----------------------

**Identifies and classifies defective detector pixels
using dark- and flat-field calibration image stacks.**

This primitive constructs a master bad pixel mask (BPM) from stacks of
calibrated detector slope images. Dark frames identify anomalous dark signals,
while flat-field frames identify abnormal pixel response. Both types of calibration
frames are also examined for temporal instability and persistent local spatial anomalies.
The routine returns a binary mask, a bitwise data-quality (DQ) map, individual
defect-category masks, and diagnostic products. Dark- and flat-derived masks
can subsequently be combined to form the master BPM used during data reduction.

Behaviour
^^^^^^^^^

* Accepts a three-dimensional stack of slope images with
  dimensions ``(n_frames, ny, nx)`` and processes it in either ``dark`` or ``flat`` mode.
* Identifies the detector reference-pixel border (four pixels wide by default).
  Reference pixels can be excluded from detection statistics and independently flagged in the DQ map.
* Flags pixels with non-finite values in any input frame.
* Detects temporally unstable pixels using robust frame-to-frame scatter after
  optional subtraction of global, row, and column common-mode offsets.
* In dark mode, detects hot and cold pixels using the median dark signal or
  a locally background-subtracted dark residual.
* In flat mode, detects low-response and high-response pixels from normalized flat-field signals.
* Detects persistent spatial outliers by comparing each frame with a local median-filtered image.
* Optionally suppresses candidate defects associated with coherent row
  structure, while retaining candidates that remain anomalous after row correction.
* Optionally flags science pixels adjacent to hot, low-response, or high-response pixels.
* Identifies connected groups of defective pixels and records their cluster membership.
* Combines all detected categories into the final science-pixel BPM.

Detection Procedure
^^^^^^^^^^^^^^^^^^^

**1. Reference and non-finite pixels**

A configurable reference border is identified using ``ref_width`` (default: 4 pixels).
If ``exclude_reference_from_stats=True``, the border is excluded from statistical detection.
A science pixel is flagged ``NONFINITE`` if any frame contains a non-finite value at that location.

**2. Temporal instability**

The algorithm optionally removes the per-frame global median, row medians,
and column medians before measuring temporal variations. For each pixel, the robust temporal scatter is

.. math::

   \sigma_{\mathrm{temp}}(x,y) = 1.4826\,\mathrm{median}_k\left|I_k'(x,y)-\mathrm{median}_j I_j'(x,y)\right|,

where :math:`I_k'` is the common-mode-corrected frame. Pixels are marked ``UNSTABLE``
when their temporal scatter exceeds either a user-supplied absolute cutoff (``temporal_sigma_abs_cut``)
or a robust population threshold:

.. math::

   T_{\mathrm{temp}} = \mathrm{median}(\sigma_{\mathrm{temp}})
   + N_{\mathrm{temp}}\,\sigma_{\mathrm{robust}}(\sigma_{\mathrm{temp}}).

Here ``temporal_sigma_thresh`` sets :math:`N_{\mathrm{temp}}`.
The robust population scatter is calculated from the median absolute deviation (MAD).

**3. Hot and cold pixels from dark frames**

The median dark image is calculated across the input stack. By default,
a local background is estimated with a median filter and subtracted before classification:

.. math::

   D_{\mathrm{res}}(x,y) = D_{\mathrm{med}}(x,y)
   - \mathrm{MedianFilter}(D_{\mathrm{med}})(x,y).

The local-background filter size is controlled by ``dark_residual_kernel_size``.
Hot and cold pixels are identified when the tested dark image exceeds
the upper or lower robust population threshold:

.. math::

   T_{\mathrm{hot}} = \widetilde{D} + N_{\mathrm{hot}}\sigma_D,
   \qquad
   T_{\mathrm{cold}} = \widetilde{D} - N_{\mathrm{cold}}\sigma_D.

Here :math:`\widetilde{D}` and :math:`\sigma_D` are the median and MAD-derived
scatter of the tested dark image. The default multipliers are ``dark_hot_sigma``
and ``dark_cold_sigma``. Alternatively, ``dark_hot_abs_cut`` and ``dark_cold_abs_cut``
specify absolute thresholds in the units of the tested dark image. Local-background
subtraction can be disabled with ``use_local_dark_residual=False``.

**4. Low- and high-response pixels from flat frames**

The median flat image is normalized either by a local median-filtered illumination
model (``normalize_flats=True``) or by the global median (``normalize_flats=False``):

.. math::

   R(x,y) = \frac{F_{\mathrm{med}}(x,y)}{F_{\mathrm{illum}}(x,y)}.

Pixels are flagged ``LOW_QE`` when :math:`R < T_{\mathrm{low}}` and ``HIGH_RESPONSE``
when :math:`R > T_{\mathrm{high}}`. The default thresholds are ``flat_low_response_thresh``
and ``flat_high_response_thresh``. The local illumination filter size is controlled
by ``flat_norm_kernel_size``.

**5. Persistent spatial outliers**

Each input frame is compared with a local median-filtered version of itself.
The local deviation and robust scatter are

.. math::

   \delta_k(x,y) = |I_k^{\mathrm{test}}(x,y)-M_k(x,y)|,

.. math::

   \sigma_{\mathrm{local},k}(x,y) = 1.4826\,
   \mathrm{MedianFilter}(\delta_k)(x,y),

where :math:`M_k` is the local median image. Flat frames can first be normalized by
their local illumination; dark frames can first have their local background removed.
A pixel is a candidate spatial outlier in frame :math:`k` if

.. math::

   \delta_k(x,y) > N_{\mathrm{spatial}}\,\sigma_{\mathrm{local},k}(x,y).

The default spatial threshold is ``spatial_sigma_thresh`` with ``spatial_kernel_size``.
A robust positive floor is used when the local scatter is zero or invalid.
Pixels are assigned ``SPATIAL_OUTLIER`` only if they satisfy the criterion in at
least ``min_frames_bad_spatial`` frames.

**6. Conservative row-artifact suppression**

When ``row_correction=True``, the detection tests are repeated after subtracting row medians
for dark images or dividing by row medians for normalized flat responses. A candidate is removed
only when it disappears after row correction **and** its row originally contained a sufficiently
large fraction of candidate defects. The fraction is set by ``row_suppression_fraction``.
This is a conservative heuristic intended to reduce false positives from
coherent row structure; the removed candidates are saved for inspection. The input calibration
images are not modified by this diagnostic correction.

**7. Adjacent pixels and clusters**

When ``neighbor_radius`` is greater than zero, a square-neighborhood dilation identifies
science pixels adjacent to ``HOT``, ``LOW_QE``, and ``HIGH_RESPONSE`` defects. These
receive ``ADJ_HOT``, ``ADJ_LOW_QE``, and ``ADJ_OPEN`` flags, respectively. The default ``neighbor_radius=0``
disables expansion beyond the original defect pixels.

Connected components are found using eight-neighbor connectivity on the union of the core
defect categories. Components with at least ``cluster_min_size`` pixels receive the
additional ``CLUSTER`` flag. Cluster membership is a classification of already-detected defects,
not a separate criterion for adding otherwise good pixels.

**8. Final mask**

All detected DQ bits are combined. The returned ``final_bpm`` is true for
flagged **science pixels only**; reference pixels are excluded from this binary mask even
when ``REFERENCE_PIXEL`` is set in the DQ map. Dark- and flat-derived final masks can be
combined using a logical OR to obtain the overall master BPM.

Data Quality (DQ) Flags
^^^^^^^^^^^^^^^^^^^^^^^

Each category is represented by an independent bit in the ``uint16`` DQ map. A pixel may
have multiple flags.

.. list-table:: Bad-pixel DQ bit definitions
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Value
     - Meaning
   * - ``NONFINITE``
     - 1
     - Non-finite value in the calibration stack.
   * - ``UNSTABLE``
     - 2
     - Excessive temporal scatter.
   * - ``HOT``
     - 4
     - Abnormally high dark signal.
   * - ``COLD``
     - 8
     - Abnormally low dark signal.
   * - ``LOW_QE``
     - 16
     - Low normalized flat response.
   * - ``HIGH_RESPONSE``
     - 32
     - High normalized flat response.
   * - ``SPATIAL_OUTLIER``
     - 64
     - Persistent local spatial anomaly.
   * - ``ADJ_HOT``
     - 128
     - Pixel adjacent to a hot pixel.
   * - ``ADJ_LOW_QE``
     - 256
     - Pixel adjacent to a low-response pixel.
   * - ``ADJ_OPEN``
     - 512
     - Pixel adjacent to a high-response pixel.
   * - ``CLUSTER``
     - 1024
     - Member of a sufficiently large connected defect cluster.
   * - ``REFERENCE_PIXEL``
     - 2048
     - Detector reference-border pixel, optionally recorded in DQ.

Output
^^^^^^

The ``generate_bpm()`` function returns:

* ``final_bpm`` — boolean two-dimensional science-pixel bad-pixel mask.
* ``dq_map`` — ``uint16`` bitwise map containing the defect classifications.
* ``masks`` — individual boolean category masks, including ``reference_pixel`` and ``science_pixel``.
* ``products`` — diagnostic arrays, including ``stack_median``, ``temporal_sigma_map``,
  ``temporal_unstable_threshold``, ``spatial_count``, and optional row-suppression products.
  Dark processing additionally records dark levels, local backgrounds, residuals, and
  hot/cold thresholds; flat processing records the normalized flat response.

The optional ``save_category_maps()`` utility writes the binary BPM, DQ map,
individual category masks, and available diagnostic products to separate FITS
files. The ``create_bpm_header()`` utility builds calibration-provenance metadata,
including camera, detector clock mode, numbers of input dark and flat slopes,
selected parameters, and input filenames.


The related code is located at:

.. code-block:: bash

   /SCALES-DRP/scalesdrp/Primitives/bpm_create.py

.. figure:: /_static/plots/bpm.png
   :width: 600px
   :align: center

   Example bad-pixel masks for the IFS detector (left) and imaging detector (right).



.. _rmatrix:

Rectification Matrix
--------------------

**Generates the detector-to-lenslet mapping required for IFS spectral extraction.**

The rectification matrix is generated from a sequence of monochromatic
calibration exposures spanning the wavelength range of each IFS observing
mode. These calibration images measure the detector response of individual
lenslet spectra as a function of wavelength and provide the basis for
both optimal and chi-square spectral extraction.

For each observing mode, the pipeline identifies monochromatic lenslet
spots, tracks their positions across wavelength, and registers the
detected spectral traces to the physical lenslet array. The measured
spot profiles are then used to construct sparse rectification matrices
describing the mapping between detector pixels and the extracted
spectral cube.

In addition to rectification matrices constructed directly from the
measured calibration exposures, the pipeline fits the monochromatic
spot profiles with two-dimensional Gaussian models and interpolates
these models onto a regularly sampled wavelength grid.

Rectification matrix generation is controlled by the
``create_rectmat`` configuration parameter. When disabled, the
primitive skips matrix generation.

Behaviour
^^^^^^^^^

* Groups the processed monochromatic calibration exposures by IFS
  observing mode and sorts them by wavelength.

* Subtracts the common background structure from each monochromatic
  calibration image.

* Optionally removes horizontal detector striping using source-masked
  row-median subtraction.

* Identifies monochromatic lenslet spots using Gaussian smoothing
  and local peak detection.

* Tracks individual lenslet spots across successive wavelengths
  using nearest-neighbour matching.

* Identifies and consolidates duplicate spectral traces and removes
  isolated or incomplete detections.

* Registers the detected traces to the physical lenslet array:

  - For low-resolution modes, fits a polynomial mapping between
    detector coordinates and lenslet-grid indices.

  - For medium-resolution modes, uses the expected lenslet geometry
    to assign the detected traces to the lenslet array.

* For low-resolution modes, predicts missing spot positions using
  the measured wavelength-dependent trace shape and individual
  lenslet-trace fits.

* Rejects predicted low-resolution spots that do not satisfy the
  required signal-to-noise threshold.

* Generates a positional calibration array (``posarr``) containing
  the detector coordinates and extraction boundaries of each
  lenslet spot at every calibration wavelength.

* Extracts monochromatic spot profiles from the calibration images
  and measures their centroids.

* Fits two-dimensional Gaussian models to the measured spot profiles
  and interpolates the fitted parameters onto a regular wavelength grid.

* Constructs sparse rectification matrices from both the measured
  monochromatic PSFs and the interpolated Gaussian PSF models.

* Generates rectification matrices for optimal and chi-square
  spectral extraction.

* Computes the associated lenslet flat-field and line-spread-function
  calibration products for optimal extraction.

* Optionally generates shifted interpolated rectification matrices
  using the configured detector x- and y-offsets.

* Saves the rectification matrices, wavelength arrays, positional
  calibration products, and diagnostic images for subsequent
  pipeline processing.


Rectification Matrix Construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rectification matrices describe the contribution of each
lenslet and wavelength element to the detector image.

For a spectral flux vector :math:`\mathbf{f}`, the forward
model of the detector image can be expressed as

.. math::

  \mathbf{d}_{\mathrm{model}}
  =
  \mathbf{R}_{\mathrm{C2}} \mathbf{f},

where :math:`\mathbf{d}_{\mathrm{model}}` is the modeled detector
image, :math:`\mathbf{f}` contains the spectral fluxes for all
lenslets and wavelength bins, and
:math:`\mathbf{R}_{\mathrm{C2}}` is the sparse forward-model
rectification matrix.

For optimal extraction, a separate rectification matrix maps
detector pixel values into the spectral flux space:

.. math::

  \mathbf{f}_{\mathrm{extract}}
  =
  \mathbf{R}_{\mathrm{OPT}} \mathbf{d},

where :math:`\mathbf{d}` is the detector image and
:math:`\mathbf{R}_{\mathrm{OPT}}` contains the extraction
weights derived from the monochromatic PSFs.

The matrices are stored in sparse format to reduce memory
requirements, since each lenslet spectrum occupies only a
small fraction of the detector.

Wavelength Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^

The measured monochromatic spot profiles are fitted using
two-dimensional Gaussian models. The fitted PSF parameters
are interpolated across wavelength to construct models
at wavelengths between the available calibration exposures.

The interpolated wavelength grid contains:

* 27 wavelength samples for low-resolution modes.

* 600 wavelength samples for medium-resolution modes.

The resulting interpolated PSFs are used to generate both
optimal and chi-square rectification matrices.

For low-resolution modes, the interpolated wavelength range
extends between the minimum and maximum wavelengths of the
available monochromatic calibration exposures.

For medium-resolution modes, the interpolation uses the
configured wavelength limits of the observing mode.

The interpolated matrices provide a more finely sampled
spectral representation than the original monochromatic
calibration sequence.

Calibration Products
^^^^^^^^^^^^^^^^^^^^

The rectification calibration produces the following
products for each IFS observing mode.

**Positional and PSF calibration products**

* ``posarr`` -- positional calibration array containing
  the spot centroid coordinates, extraction boundaries,
  and measured spot intensities for each wavelength
  and lenslet.

* ``cropped_spotpsf_arr`` -- cropped monochromatic PSF
  images associated with the detected lenslet spots.

* ``centroids`` -- measured centroids of the
  monochromatic spot profiles.

* ``lams`` -- wavelengths of the input monochromatic
  calibration exposures.

* ``gauss_spotpars`` -- fitted Gaussian PSF parameters
  at the measured calibration wavelengths.

* ``gauss_spotpars_interpd`` -- interpolated Gaussian
  PSF parameters on the regularly sampled wavelength grid.

* ``intp_lams`` -- wavelength grid used for the
  interpolated rectification matrices.

**Rectification matrices**

* ``OPT_rectmat`` -- sparse rectification matrix
  constructed from the measured monochromatic PSFs
  for optimal spectral extraction.

* ``OPT_intp_rectmat`` -- sparse optimal-extraction
  rectification matrix constructed from interpolated
  Gaussian PSF models.

* ``C2_rectmat`` -- sparse forward-model rectification
  matrix constructed from measured monochromatic PSFs
  for chi-square spectral extraction.

* ``C2_intp_rectmat`` -- sparse forward-model
  rectification matrix constructed from interpolated
  Gaussian PSF models.

**Additional calibration products**

* ``OPT_lsf`` -- line-spread-function calibration
  derived from the measured optimal-extraction matrix.

* ``OPT_lflat`` -- lenslet flat-field calibration
  derived from the measured optimal-extraction matrix.

* ``OPT_intp_lsf`` -- line-spread-function calibration
  derived from the interpolated optimal-extraction matrix.

* ``OPT_intp_lflat`` -- lenslet flat-field calibration
  derived from the interpolated optimal-extraction matrix.

When non-zero detector offsets are specified, additional
shifted versions of the interpolated optimal and chi-square
rectification matrices are generated.

The pipeline also saves Gaussian PSF model images,
residual images, and centroid diagnostic plots to evaluate
the quality of the calibration.

Output
^^^^^^

The generated rectification matrices are saved as compressed
sparse matrix files (``.npz``), while the associated
positional, wavelength, PSF, and extraction calibration
products are saved as FITS files.

The rectification matrices are stored separately for each
IFS observing mode and can be loaded during subsequent
spectral extraction without repeating the monochromatic
calibration procedure.

The related code can be found here:

.. code-block:: bash

  /SCALES-DRP/scalesdrp/Primitives/ProcessMonochrom.py




Master Calibration Frame Generation
-----------------------------------

**Combines multiple calibration exposures into high signal-to-noise
master calibration frames.**


This primitive combines multiple calibration exposures to generate
master calibration frames used throughout the SCALES-DRP. It supports
the creation of master bias, dark, and detector flat frames, as well
as master monochromatic calibration frames at individual wavelengths,
by combining calibrated images while rejecting transient artifacts
and propagating uncertainties.


Input calibration frames can be provided either as a three-dimensional
image stack or as a list of two-dimensional images. When available,
the corresponding uncertainty images are used for inverse-variance
weighted combination.

Behaviour
^^^^^^^^^

* Accepts multiple calibration images with optional uncertainty maps.

* Optionally scales individual frames before combination, such as for
  exposure-time or illumination normalization.

* Excludes non-finite pixel values and invalid uncertainties from
  the combination.

* Applies iterative sigma clipping to identify and reject transient
  artifacts and statistical outliers.

* Supports three combination methods:

  - ``ivw`` — inverse-variance weighted mean.
  - ``mean`` — arithmetic mean.
  - ``median`` — median combination.

* Estimates and propagates the uncertainty of the combined master
  calibration frame according to the selected combination method.

* Identifies output pixels with insufficient valid input samples
  and marks them as invalid.

* Optionally returns the final inlier mask, indicating which input
  frames contributed to each pixel in the master calibration frame.

Output
^^^^^^

The primitive returns:

* **Master calibration frame** — a two-dimensional image generated
  by combining the input calibration exposures.

* **Uncertainty map** — the estimated uncertainty associated with
  each pixel in the master calibration frame.

* **Inlier mask (optional)** — a mask identifying the input samples
  retained after sigma clipping.

The corresponding implementation can be found in:

.. code-block:: text

   SCALES-DRP/scalesdrp/Primitives/scales_basic.py
