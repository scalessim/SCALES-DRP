import pandas as pd
import numpy as np
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
from scipy.optimize import minimize
import astropy.io.fits as pyfits
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import scalesdrp.primitives.robust as robust
from scipy.optimize import leastsq
from scipy.signal import savgol_filter



def reffix_hxrg(
    cube,
    nchans=4,
    in_place=False,

    # Amplifier/top-bottom reference correction
    altcol=False,
    channelwise=True,
    supermean=False,
    top_ref=True,
    bot_ref=True,
    ntop=4,
    nbot=4,
    amp_mean_func=robust.mean,

    # Optional row-dependent ACN correction
    do_acn=True,
    acn_avg_type='pix', #frame, pix
    acn_perint=False,
    acn_edge_wrap=False,
    acn_left_ref=True,
    acn_right_ref=True,
    acn_nleft=4,
    acn_nright=4,
    acn_mean_func=np.median,
    acn_smooth=True,
    acn_savgol=False,
    acn_winsize=31,
    acn_order=3,

    # Optional residual per-column correction
    resid_colsub=False,

    # 1/f correction
    fixcol=False,
    ref_avg_type='row_wise', #frame, pix, row_wise
    ref_perint=False,
    ref_edge_wrap=False,
    ref_left=True,
    ref_right=True,
    ref_nleft=4,
    ref_nright=4,
    ref_mean_func=np.median,
    ref_smooth=True,
    ref_savgol=False,
    ref_winsize=31,
    ref_order=3,

    #pickup noise removal
    pickup=True,
    sigma_thresh=4.0,
    dilate_iter=2,
    highpass_size=101,
    per_amp=False,

    **kwargs
):
    """Apply HxRG reference-pixel corrections.

    Order: amplifier correction -> optional ACN -> optional residual-column
    correction -> optional side-reference 1/f correction.

    ``altcol=True`` gives separate even/odd scalar offsets per amplifier/read.
    ``altcol=False, channelwise=True`` gives one scalar per amplifier/read.
    ``altcol=False, channelwise=False`` gives one value per column/read.
    """
    arr = np.asarray(cube)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=True)
    elif not in_place:
        arr = arr.copy()

    arr = reffix_amps(
        arr, nchans=nchans, in_place=False,
        altcol=altcol, channelwise=channelwise,
        supermean=supermean, top_ref=top_ref, bot_ref=bot_ref,
        ntop=ntop, nbot=nbot, mean_func=amp_mean_func,
    )

    if do_acn:
        arr = acn_filter(
            arr, in_place=False,
            avg_type=acn_avg_type, perint=acn_perint,
            edge_wrap=acn_edge_wrap,
            left_ref=acn_left_ref, right_ref=acn_right_ref,
            nleft=acn_nleft, nright=acn_nright,
            mean_func=acn_mean_func, smooth=acn_smooth,
            savgol=acn_savgol, winsize=acn_winsize, order=acn_order,
        )

    if resid_colsub:
        arr = sub_resid_col(
            arr, nchans=nchans, in_place=False,
            top_ref=top_ref, bot_ref=bot_ref, ntop=ntop, nbot=nbot,
        )

    if fixcol:
        #arr = ref_filter_orig(
        #    arr, nchans=nchans, in_place=False,
        #    avg_type=ref_avg_type, perint=ref_perint,
        #    edge_wrap=ref_edge_wrap,
        #    left_ref=ref_left, right_ref=ref_right,
        #    nleft=ref_nleft, nright=ref_nright,
        #    mean_func=ref_mean_func, smooth=ref_smooth,
        #    savgol=ref_savgol, winsize=ref_winsize, order=ref_order,
        #)
        arr = ref_filter(
            arr,
            nchans=nchans,
            in_place=False,
            avg_type=ref_avg_type,
            perint=ref_perint,
            edge_wrap=ref_edge_wrap,
            left_ref=ref_left,
            right_ref=ref_right,
            nleft=ref_nleft,
            nright=ref_nright,
            mean_func=ref_mean_func,
            smooth=ref_smooth,
            savgol=ref_savgol,
            winsize=ref_winsize,
            order=ref_order,
            method="direct",
            per_amp=True,
        )
    if pickup:
        arr = correct_evolving_row_pickup(
            arr,
            nleft=4,
            nright=4,
            ntop=4,
            nbot=4,
            sigma_thresh=sigma_thresh,
            dilate_iter=dilate_iter,
            min_good_fraction=0.25,
            highpass_size=highpass_size,
            remove_full_row_pattern=True,
            nchans=nchans,
            per_amp=per_amp,
            n_passes=1,
            return_model=False,
        )

    return arr

################################## acn ################################
def reffix_amps(
    cube,
    nchans=4,
    in_place=True,
    altcol=False,
    channelwise=True,
    supermean=False,
    top_ref=True,
    bot_ref=True,
    ntop=4,
    nbot=4,
    mean_func=robust.mean,
    **kwargs
):
    """Correct amplifier offsets using top/bottom reference rows.

    Modes
    -----
    altcol=True
        Separate even/odd scalar offsets per amplifier/read.
    altcol=False, channelwise=True
        One scalar offset per amplifier/read.
    altcol=False, channelwise=False
        One reference-derived offset per detector column/read.
    """
    arr = np.asarray(cube)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=True)
    elif not in_place:
        arr = arr.copy()

    single_frame = arr.ndim == 2
    if single_frame:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Input data can only have 2 or 3 dimensions. Found {arr.ndim}.")

    nz, ny, nx = arr.shape
    if nx % nchans != 0:
        raise ValueError(f"Detector width {nx} is not divisible by nchans={nchans}.")
    chsize = nx // nchans

    nt = ntop if top_ref else 0
    nb = nbot if bot_ref else 0
    if nt < 0 or nb < 0:
        raise ValueError("ntop and nbot must be non-negative.")
    if nt + nb == 0:
        return arr[0] if single_frame else arr

    blocks=[]
    if nb > 0:
        blocks.append(arr[:, :nb, :])
    if nt > 0:
        blocks.append(arr[:, -nt:, :])
    refs_all=np.concatenate(blocks, axis=1)

    smean = mean_func(refs_all) if supermean else 0.0
    refs_amps_avg = calc_avg_amps(
        refs_all, arr.shape, nchans=nchans,
        altcol=altcol, channelwise=channelwise,
        mean_func=mean_func,
    )

    for ch in range(nchans):
        x0=ch*chsize
        x1=x0+chsize
        if altcol:
            arr[:, :, x0:x1:2] -= refs_amps_avg[0][ch][:, None, None]
            arr[:, :, x0+1:x1:2] -= refs_amps_avg[1][ch][:, None, None]
        elif channelwise:
            arr[:, :, x0:x1] -= refs_amps_avg[ch][:, None, None]
        else:
            arr[:, :, x0:x1] -= refs_amps_avg[ch][:, None, :]

    if supermean:
        arr += smean
    return arr[0] if single_frame else arr


def calc_avg_amps(
    refs_all,
    data_shape,
    nchans=4,
    altcol=False,
    channelwise=True,
    mean_func=robust.mean,
    **kwargs
):
    """Calculate reference offsets used by ``reffix_amps``."""
    nz_ref, nref, nx = refs_all.shape
    nz, ny, nx_full = data_shape
    if nx != nx_full or nz_ref != nz:
        raise ValueError("refs_all and data_shape are inconsistent.")
    if nx % nchans != 0:
        raise ValueError(f"Detector width {nx} is not divisible by nchans={nchans}.")

    chsize=nx//nchans

    if altcol:
        even=np.empty((nchans,nz), dtype=np.float32)
        odd=np.empty((nchans,nz), dtype=np.float32)
        for ch in range(nchans):
            x0=ch*chsize; x1=x0+chsize
            even[ch]=mean_func(refs_all[:,:,x0:x1:2].reshape(nz,-1), axis=1)
            odd[ch]=mean_func(refs_all[:,:,x0+1:x1:2].reshape(nz,-1), axis=1)
        return even, odd

    if channelwise:
        offsets=np.empty((nchans,nz), dtype=np.float32)
        for ch in range(nchans):
            x0=ch*chsize; x1=x0+chsize
            offsets[ch]=mean_func(refs_all[:,:,x0:x1].reshape(nz,-1), axis=1)
        return offsets

    offsets=np.empty((nchans,nz,chsize), dtype=np.float32)
    for ch in range(nchans):
        x0=ch*chsize; x1=x0+chsize
        offsets[ch]=mean_func(refs_all[:,:,x0:x1], axis=1).astype(np.float32)
    return offsets

def sub_resid_col(cube, nchans=4, in_place=True,
    top_ref=True, bot_ref=True, ntop=4, nbot=4, **kwargs):

    """
    Parameters
    ----------
    cube : ndarray
        Input datacube. Can be two or three dimensions (nz,ny,nx).
    nchans : int
        Number of output amplifier channels in the detector. Default=4.
    in_place : bool
        Perform calculations in place. Input array is overwritten.
    top_ref : bool
        Include top reference rows when correcting channel offsets.
    bot_ref : bool
        Include bottom reference rows when correcting channel offsets.
    ntop : int
        Specify the number of top reference rows.
    nbot : int
        Specify the number of bottom reference rows.
    Keyword Args
    ------------
    mean_func : func
        Function used to calculate averages.
    """
    if not np.issubdtype(cube.dtype, np.floating):
        cube = cube.astype(np.float32, copy=not in_place)
        in_place = True

    if not in_place:
        cube = np.copy(cube)
    ndim = len(cube.shape)
    if ndim==2:
        ny,nx = cube.shape
        nz = 1
        cube = cube.reshape((nz,ny,nx))
    elif ndim==3:
        nz, ny, nx = cube.shape
    else:
        raise ValueError('Input data can only have 2 or 3 dimensions.Found {} dimensions.'.format(ndim))

    chsize = int(nx / nchans)
    # Number of reference rows to use
    # Set nt or nb equal to 0 if we don't want to use either
    nt = ntop if top_ref else 0
    nb = nbot if bot_ref else 0
    if (nt+nb)==0:
        print("No reference pixels available for use. Returning...")
        return
    # Slice out reference pixels
    refs_bot = cube[:,:nb,:]
    refs_top = cube[:,-nt:,:]
    if nt==0:
        refs_all = refs_bot
    elif nb==0:
        refs_all = refs_top
    else:
        refs_all = np.hstack((refs_bot, refs_top))
    assert refs_all.shape[1] == (nb+nt)

    refs_amps_avg = np.mean(refs_all,axis=1)
    for i in range(len(cube)):
        for j in range(len(cube[i])):
            cube[i,j]-=refs_amps_avg[i]
    return cube


############# additional acn ###################################

def acn_filter(cube,
               in_place=True,
               avg_type='pix',
               perint=False,
               edge_wrap=False,
               left_ref=True,
               right_ref=True,
               nleft=4,
               nright=4,
               mean_func=np.median,
               smooth=True,
               **kwargs):
    """
    Row-dependent differences between even and odd columns
    (i.e. ACN that changes with row, not just a constant offset per column).
    They are row-series describing how the even or odd reference pixels
    vary vs row (and frame), after baseline removal.

    ACN correction using side reference columns.
    Separately estimates row-dependent even and odd column offsets,
    then subtracts them from the full image.

    Parameters
    ----------
    cube : ndarray
        Input data:
          - (H, W)  single frame, or
          - (N, H, W) stack of frames.
    in_place : bool
        If False, the input array will be copied.
    avg_type : {'pix', 'frame', 'int'}
        Baseline removal mode for reference pixels:
          - 'pix'   : remove per-pixel mean over integration (best for many frames).
          - 'frame' : remove per-frame global mean of refs.
          - 'int'   : remove a single global mean over the whole ramp.
    perint : bool
        Passed to `calc_col_smooth`: smooth per integration instead of per frame.
    edge_wrap : bool
        Passed to `calc_col_smooth`: mirror edges before smoothing to reduce ringing.
    left_ref, right_ref : bool
        Whether to use left and/or right side reference columns.
    nleft, nright : int
        Number of left/right reference columns to use.
    mean_func : callable
        Function to compute averages (e.g., np.median or robust.mean).

    Returns
    -------
    out : ndarray
        ACN-corrected array (same shape as input, float32).
    """
    arr = np.asarray(cube)
    print(arr.shape)
    if arr.ndim == 2:
        single_frame = True
        arr = arr[np.newaxis, ...]   # (1, H, W)
    elif arr.ndim == 3:
        single_frame = False
    else:
        raise ValueError(f"acn_filter: input must be 2D or 3D, got shape {arr.shape}")

    if not in_place:
        arr = np.copy(arr)

    N, H, W = arr.shape

    # Decide how many side references we actually use
    nl = nleft  if left_ref  else 0
    nr = nright if right_ref else 0

    if nl < 0 or nr < 0:
        raise ValueError("nleft and nright must be non-negative.")
    if (nl + nr) == 0:
        print("acn_filter: No side reference columns enabled. Returning input.")
        return cube

    out = arr.astype(np.float32, copy=True)

    # Slice side reference columns
    refs_left  = out[:, :, :nl]   if nl > 0 else None  # (N, H, nl)
    refs_right = out[:, :, -nr:]  if nr > 0 else None  # (N, H, nr)

    def _normalize_and_avg(refs_left, refs_right, avg_type, mean_func):
        """
        Given left/right refs for even or odd,
        perform avg_type baseline removal and then average along columns
        to get (N, H) row-wise reference values.
        """
        nl_flag = 0 if refs_left  is None else 1
        nr_flag = 0 if refs_right is None else 1

        if nl_flag == 0 and nr_flag == 0:
            # No refs for this parity at all; return zeros
            return np.zeros((N, H), dtype=np.float32)

        # Make copies so we don't mutate outer refs
        if nl_flag:
            refs_left = np.copy(refs_left)
            Nloc, Hloc, Cleft = refs_left.shape
        if nr_flag:
            refs_right = np.copy(refs_right)
            if not nl_flag:
                Nloc, Hloc, Cright = refs_right.shape
            else:
                _, _, Cright = refs_right.shape

        if avg_type is None:
            mode = 'frame'
        else:
            mode = avg_type

        if Nloc == 1:
            mode = 'int'  # 'int' == 'frame' when only one frame

        # ---- Remove intrinsic offsets depending on mode ----
        if 'int' in mode:
            # One global scalar over the integration
            #compute single scalar value over frames and all left and right reference
            if nl_flag:
                refs_left  -= mean_func(refs_left)
            if nr_flag:
                refs_right -= mean_func(refs_right)

        elif 'frame' in mode:
            # One scalar per frame
            if nl_flag:
                rl_flat = refs_left.reshape(Nloc, -1)
                rl_mean = mean_func(rl_flat, axis=1)
                plt.figure()
                plt.title('left')
                plt.plot(rl_mean)
                plt.show()
                for i in range(Nloc):
                    refs_left[i] -= rl_mean[i]
            if nr_flag:
                rr_flat = refs_right.reshape(Nloc, -1)
                rr_mean = mean_func(rr_flat, axis=1)
                plt.figure()
                plt.title('right')
                plt.plot(rr_mean)
                plt.show()
                for i in range(Nloc):
                    refs_right[i] -= rr_mean[i]

        elif 'pix' in mode:
            # Per-pixel mean over frames
            if nl_flag:
                rl_mean = mean_func(refs_left, axis=0)   # (H, Cleft)
                #print(rl_mean.shape)
                #print(np.where(np.isnan(rl_mean)==True))
                #plt.figure()
                #plt.plot(rl_mean)
                #plt.show()
                for i in range(Nloc):
                    refs_left[i] -= rl_mean
                    #plt.figure()
                    #plt.title('acn using left side ref')
                    #plt.plot(refs_left[i])
                #plt.show()
            if nr_flag:
                rr_mean = mean_func(refs_right, axis=0)  # (H, Cright)
                #print(np.where(np.isnan(rr_mean)==True))
                #plt.figure()
                #plt.plot(rr_mean)
                #plt.show()
                for i in range(Nloc):
                    refs_right[i] -= rr_mean
                    #plt.figure()
                    #plt.title('acn using right side ref')
                    #plt.plot(refs_left[i])
                #plt.show()

        # ---- Average left/right columns down to a single value per row ----
        if nl_flag == 0:
            refs_side_avg = refs_right.mean(axis=2)      # (N, H)
        elif nr_flag == 0:
            refs_side_avg = refs_left.mean(axis=2)       # (N, H)
        else:
            # Average left and right
            refs_side_avg = (refs_left.mean(axis=2) + refs_right.mean(axis=2)) / 2.0

        return refs_side_avg.astype(np.float32)

    # --- Split side refs into even/odd columns

    # Global column indices
    cols = np.arange(W) #W is 4

    # Left side global indices: 0 .. nl-1
    if nl > 0:
        left_cols = cols[:nl]
        left_even_mask = (left_cols % 2) == 0
        left_odd_mask  = ~left_even_mask
        refs_left_even = refs_left[:, :, left_even_mask] if left_even_mask.any() else None
        refs_left_odd  = refs_left[:, :, left_odd_mask]  if left_odd_mask.any()  else None
    else:
        refs_left_even = refs_left_odd = None

    # Right side global indices: W-nr .. W-1
    if nr > 0:
        right_cols = cols[-nr:]
        right_even_mask = (right_cols % 2) == 0
        right_odd_mask  = ~right_even_mask
        refs_right_even = refs_right[:, :, right_even_mask] if right_even_mask.any() else None
        refs_right_odd  = refs_right[:, :, right_odd_mask]  if right_odd_mask.any()  else None
    else:
        refs_right_even = refs_right_odd = None

    # --- Compute raw row-wise ACN refs for even and odd separately ---

    # Shape (N, H) each
    ref_even = _normalize_and_avg(refs_left_even, refs_right_even, avg_type, mean_func) #(2,2048)
    ref_odd  = _normalize_and_avg(refs_left_odd,  refs_right_odd,  avg_type, mean_func) #(2,2048)

    # --- Optional smoothing of row-wise ACN reference series ---

    if smooth:
        ref_even_sm = calc_col_smooth(
            ref_even, out.shape, perint=perint, edge_wrap=edge_wrap,
            delt=kwargs.get('delt', 5.24e-4),
            savgol=kwargs.get('savgol', False),
            winsize=kwargs.get('winsize', 31),
            order=kwargs.get('order', 3),
        )
        ref_odd_sm = calc_col_smooth(
            ref_odd, out.shape, perint=perint, edge_wrap=edge_wrap,
            delt=kwargs.get('delt', 5.24e-4),
            savgol=kwargs.get('savgol', False),
            winsize=kwargs.get('winsize', 31),
            order=kwargs.get('order', 3),
        )
    else:
        ref_even_sm = ref_even
        ref_odd_sm = ref_odd

    # --- Subtract from even/odd columns across the entire image ---
    #Each row y gets: one subtraction for even columns (ref_even_sm[:, y]),
    #one subtraction for odd columns (ref_odd_sm[:, y]).

    even_cols = cols[cols % 2 == 0]
    odd_cols  = cols[cols % 2 == 1]

    if even_cols.size > 0:
        out[:, :, even_cols] -= ref_even_sm.reshape(N, H, 1)
    if odd_cols.size > 0:
        out[:, :, odd_cols]  -= ref_odd_sm.reshape(N, H, 1)

    if single_frame:
        return out[0]
    return out

import numpy as np


def ref_filter_orig(
    cube,
    nchans=4,
    in_place=True,
    avg_type='row_wise',
    perint=False,
    edge_wrap=False,
    left_ref=True,
    right_ref=True,
    nleft=4,
    nright=4,
    mean_func=np.median,
    smooth=False,
    savgol=False,
    winsize=31,
    order=3,
    **kwargs
):
    """Correct horizontal 1/f stripes using side reference columns."""
    arr=np.asarray(cube)
    if not np.issubdtype(arr.dtype, np.floating):
        arr=arr.astype(np.float32, copy=True)
    elif not in_place:
        arr=arr.copy()

    single_frame=arr.ndim==2
    if single_frame:
        arr=arr[np.newaxis,...]
    elif arr.ndim!=3:
        raise ValueError(f"Input data can only have 2 or 3 dimensions. Found {arr.ndim}.")

    nz,ny,nx=arr.shape
    nl=nleft if left_ref else 0
    nr=nright if right_ref else 0
    if nl < 0 or nr < 0:
        raise ValueError("nleft and nright must be non-negative.")
    if nl+nr==0:
        return arr[0] if single_frame else arr

    refs_left=arr[:,:,:nl] if nl>0 else None
    refs_right=arr[:,:,-nr:] if nr>0 else None
    refvals=calc_avg_cols(
        refs_left, refs_right,
        avg_type=avg_type,
        mean_func=mean_func,
    )

    if smooth:
        delt=10e-6*(nx/nchans+12.0)
        model=calc_col_smooth(
            refvals, arr.shape,
            perint=perint, edge_wrap=edge_wrap,
            delt=delt, savgol=savgol,
            winsize=winsize, order=order,
        )
    else:
        model=refvals

    arr -= model.reshape(nz,ny,1)
    return arr[0] if single_frame else arr



def calc_avg_cols(
    refs_left=None,
    refs_right=None,
    avg_type='pix',
    mean_func=np.median,
    **kwargs
):
    """
    Calculate row-wise reference signal.

    avg_type
    --------
    'pix'
        Remove each reference pixel's median over the ramp.

    'frame'
        Remove one scalar reference level per frame.

    'int'
        Remove one scalar level for the whole integration.

    'row_wise'
        For every read and detector row, take the median of all
        available side reference pixels, then remove only the
        median of that row-profile for that read.
    """

    nl = 0 if refs_left is None else 1
    nr = 0 if refs_right is None else 1

    if nl == 0 and nr == 0:
        raise ValueError("No side reference pixels supplied.")

    if nl:
        refs_left = np.copy(refs_left)

    if nr:
        refs_right = np.copy(refs_right)

    if refs_left is not None:
        nz, ny, _ = refs_left.shape
    else:
        nz, ny, _ = refs_right.shape

    if avg_type is None:
        avg_type = 'frame'

    # ---------------------------------------------------------
    # NEW: direct row-wise reference estimate
    # ---------------------------------------------------------
    if 'row' in avg_type.lower():

        if nl and nr:
            # Shape: (nz, ny, nleft+nright)
            refs_all = np.concatenate(
                [refs_left, refs_right],
                axis=2
            )

        elif nl:
            refs_all = refs_left

        else:
            refs_all = refs_right

        # One value for every read and detector row
        # Shape: (nz, ny)
        refs_side_avg = np.nanmedian(
            refs_all,
            axis=2
        )

        # Remove only the overall DC level of each read.
        # Preserve ALL row-dependent structure.
        refs_side_avg -= np.nanmedian(
            refs_side_avg,
            axis=1,
            keepdims=True
        )

        return refs_side_avg.astype(np.float32)

    # ---------------------------------------------------------
    # Existing modes
    # ---------------------------------------------------------

    # Only force single-frame data to 'int' for the old modes.
    if nz == 1:
        avg_type = 'int'

    if 'int' in avg_type:

        if nl:
            refs_left -= mean_func(refs_left)

        if nr:
            refs_right -= mean_func(refs_right)

    elif 'frame' in avg_type:

        if nl:
            refs_left_mean = mean_func(
                refs_left.reshape((nz, -1)),
                axis=1
            )

            for i in range(nz):
                refs_left[i] -= refs_left_mean[i]

        if nr:
            refs_right_mean = mean_func(
                refs_right.reshape((nz, -1)),
                axis=1
            )

            for i in range(nz):
                refs_right[i] -= refs_right_mean[i]

    elif 'pix' in avg_type:

        if nl:
            refs_left_mean = mean_func(
                refs_left,
                axis=0
            )

        if nr:
            refs_right_mean = mean_func(
                refs_right,
                axis=0
            )

        for i in range(nz):

            if nl:
                refs_left[i] -= refs_left_mean

            if nr:
                refs_right[i] -= refs_right_mean

    else:
        raise ValueError(
            f"Unknown avg_type='{avg_type}'. "
            "Use 'pix', 'frame', 'int', or 'row_wise'."
        )

    # ---------------------------------------------------------
    # Collapse side references for old modes
    # ---------------------------------------------------------

    if nl and nr:

        refs_all = np.concatenate(
            [refs_left, refs_right],
            axis=2
        )

        refs_side_avg = np.nanmedian(
            refs_all,
            axis=2
        )

    elif nl:

        refs_side_avg = np.nanmedian(
            refs_left,
            axis=2
        )

    else:

        refs_side_avg = np.nanmedian(
            refs_right,
            axis=2
        )

    return refs_side_avg.astype(np.float32)


def ref_filter(
    cube,
    nchans=4,
    in_place=True,

    # Reference profile construction
    avg_type="row_wise",
    left_ref=True,
    right_ref=True,
    nleft=4,
    nright=4,
    mean_func=np.median,

    # Correction method
    method="direct",          # "direct" or "transfer"

    # Existing optional smoothing for direct mode
    smooth=False,
    perint=False,
    edge_wrap=False,
    savgol=False,
    winsize=31,
    order=3,

    # Direct-mode scale
    scale=1.0,

    # Transfer-function mode
    transfer_func=None,
    coherence=None,
    coherence_min=0.2,
    coherence_weight=False,

    # Apply independently to amplifier channels
    per_amp=False,

    # Preserve side reference pixels
    correct_ref_pixels=False,

    return_model=False,
    **kwargs,
):
    """
    Correct horizontal 1/f / pickup stripes using side reference pixels.

    Parameters
    ----------
    cube : ndarray
        Input image or ramp cube:
            (ny, nx)
            (nread, ny, nx)

    nchans : int
        Number of detector amplifier channels.

    avg_type : str
        Method passed to ``calc_avg_cols``.
        For SCALES pickup correction, ``"row_wise"`` is recommended.

    left_ref, right_ref : bool
        Use left/right side reference columns.

    nleft, nright : int
        Number of side reference columns.

    mean_func : callable
        Statistic used when constructing reference profiles.

    method : {"direct", "transfer"}
        direct
            Subtract the reference row profile directly, optionally
            after smoothing.

        transfer
            Fourier-transform the row reference profile, multiply by
            a calibrated science/reference transfer function H(f),
            and inverse-transform to obtain the predicted science
            pickup.

    smooth : bool
        Apply the existing calc_col_smooth() method.
        Used only for ``method="direct"``.

    scale : float or ndarray
        Multiplicative scale for direct mode.

        Can be:
            scalar
            (nread,)
            (nchans,)
            (nread, nchans)

    transfer_func : ndarray
        Frequency-dependent science/reference coupling.

        Supported shapes:

            (nfreq,)
                same transfer function for all reads/channels

            (nchans, nfreq)
                one transfer function per amplifier

        where

            nfreq = ny // 2 + 1

    coherence : ndarray or None
        Optional coherence corresponding to ``transfer_func``.

        Supported shapes match transfer_func.

    coherence_min : float
        Frequencies below this coherence are rejected when
        ``coherence_weight=False``.

    coherence_weight : bool
        If True, weight the transfer function continuously by
        coherence instead of using a hard cutoff.

    per_amp : bool
        If True, apply an independent transfer function to each
        amplifier.

        This should normally be True when using:
            transfer_func.shape == (nchans, nfreq)

    correct_ref_pixels : bool
        If False, leave side reference columns untouched.

    return_model : bool
        If True, also return the modeled pickup image/cube.

    Returns
    -------
    corrected : ndarray

    model : ndarray, optional
        Returned when ``return_model=True``.
    """

    # -------------------------------------------------------------
    # Prepare input
    # -------------------------------------------------------------
    arr = np.asarray(cube)

    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=True)
    elif not in_place:
        arr = arr.copy()

    single_frame = arr.ndim == 2

    if single_frame:
        arr = arr[np.newaxis, ...]

    elif arr.ndim != 3:
        raise ValueError(
            "Input data must be 2D or 3D. "
            f"Found shape {arr.shape}."
        )

    nz, ny, nx = arr.shape

    if nx % nchans != 0:
        raise ValueError(
            f"nx={nx} is not divisible by nchans={nchans}."
        )

    chsize = nx // nchans

    nl = nleft if left_ref else 0
    nr = nright if right_ref else 0

    if nl < 0 or nr < 0:
        raise ValueError(
            "nleft and nright must be non-negative."
        )

    if (nl + nr) == 0:
        result = arr[0] if single_frame else arr

        if return_model:
            return result, np.zeros_like(result)

        return result

    # -------------------------------------------------------------
    # Extract side reference pixels
    # -------------------------------------------------------------
    refs_left = (
        arr[:, :, :nl]
        if nl > 0
        else None
    )

    refs_right = (
        arr[:, :, -nr:]
        if nr > 0
        else None
    )

    # Shape:
    #     (nz, ny)
    refvals = calc_avg_cols(
        refs_left,
        refs_right,
        avg_type=avg_type,
        mean_func=mean_func,
    )

    refvals = np.asarray(
        refvals,
        dtype=np.float64,
    )

    if refvals.shape != (nz, ny):
        raise ValueError(
            "calc_avg_cols returned unexpected shape "
            f"{refvals.shape}; expected {(nz, ny)}."
        )

    # -------------------------------------------------------------
    # Output model
    # -------------------------------------------------------------
    model_cube = np.zeros_like(
        arr,
        dtype=np.float64,
    )

    method = method.lower()

    # =============================================================
    # DIRECT REFERENCE SUBTRACTION
    # =============================================================
    if method == "direct":

        if smooth:

            delt = 10e-6 * (
                nx / nchans + 12.0
            )

            row_model = calc_col_smooth(
                refvals,
                arr.shape,
                perint=perint,
                edge_wrap=edge_wrap,
                delt=delt,
                savgol=savgol,
                winsize=winsize,
                order=order,
            )

        else:

            row_model = refvals.copy()

        # ---------------------------------------------------------
        # Scale handling
        # ---------------------------------------------------------
        scale_arr = np.asarray(
            scale,
            dtype=float,
        )

        if not per_amp:

            # scalar
            if scale_arr.ndim == 0:

                scaled = (
                    row_model
                    * float(scale_arr)
                )

            # one scale per read
            elif scale_arr.shape == (nz,):

                scaled = (
                    row_model
                    * scale_arr[:, None]
                )

            else:

                raise ValueError(
                    "For per_amp=False, scale must be "
                    "a scalar or shape (nread,)."
                )

            # Apply same row model to full science region
            x0 = 0 if correct_ref_pixels else nl
            x1 = nx if correct_ref_pixels else nx - nr

            model_cube[
                :,
                :,
                x0:x1
            ] = scaled[:, :, None]

        else:

            # -----------------------------------------------------
            # amplifier-specific direct scaling
            # -----------------------------------------------------

            if scale_arr.ndim == 0:

                scale_arr = np.full(
                    (nz, nchans),
                    float(scale_arr),
                )

            elif scale_arr.shape == (nchans,):

                scale_arr = np.broadcast_to(
                    scale_arr[None, :],
                    (nz, nchans),
                )

            elif scale_arr.shape != (nz, nchans):

                raise ValueError(
                    "For per_amp=True, scale must be "
                    "scalar, (nchans,), or "
                    "(nread, nchans)."
                )

            for amp in range(nchans):

                a0 = amp * chsize
                a1 = (amp + 1) * chsize

                x0 = a0
                x1 = a1

                if not correct_ref_pixels:

                    if amp == 0:
                        x0 = max(x0, nl)

                    if amp == nchans - 1:
                        x1 = min(
                            x1,
                            nx - nr,
                        )

                scaled = (
                    row_model
                    * scale_arr[:, amp, None]
                )

                model_cube[
                    :,
                    :,
                    x0:x1
                ] = scaled[:, :, None]

    # =============================================================
    # FREQUENCY-DEPENDENT TRANSFER CORRECTION
    # =============================================================
    elif method == "transfer":

        if transfer_func is None:
            raise ValueError(
                "transfer_func must be supplied "
                "for method='transfer'."
            )

        H = np.asarray(
            transfer_func,
        )

        nfreq = ny // 2 + 1

        # ---------------------------------------------------------
        # Validate transfer function
        # ---------------------------------------------------------
        if H.ndim == 1:

            if H.shape[0] != nfreq:
                raise ValueError(
                    "transfer_func has incorrect frequency length. "
                    f"Expected {nfreq}, got {H.shape[0]}."
                )

            H = np.broadcast_to(
                H[None, :],
                (nchans, nfreq),
            )

        elif H.ndim == 2:

            if H.shape != (nchans, nfreq):
                raise ValueError(
                    "2D transfer_func must have shape "
                    f"({nchans}, {nfreq}); got {H.shape}."
                )

        else:

            raise ValueError(
                "transfer_func must have shape "
                "(nfreq,) or (nchans, nfreq)."
            )

        # ---------------------------------------------------------
        # Coherence handling
        # ---------------------------------------------------------
        if coherence is not None:

            C = np.asarray(
                coherence,
                dtype=float,
            )

            if C.ndim == 1:

                if C.shape[0] != nfreq:
                    raise ValueError(
                        "coherence has incorrect frequency length."
                    )

                C = np.broadcast_to(
                    C[None, :],
                    (nchans, nfreq),
                )

            elif C.shape != (nchans, nfreq):

                raise ValueError(
                    "coherence must match transfer_func shape."
                )

        else:

            C = None

        # ---------------------------------------------------------
        # FFT of reference profile
        # ---------------------------------------------------------
        #
        # ref_fft shape:
        #     (nz, nfreq)
        #
        ref_fft = np.fft.rfft(
            refvals,
            axis=1,
        )

        # ---------------------------------------------------------
        # Build model amplifier by amplifier
        # ---------------------------------------------------------
        for amp in range(nchans):

            H_amp = H[amp].copy()

            if C is not None:

                C_amp = C[amp]

                if coherence_weight:

                    # Smooth weighting:
                    #
                    # coherence=1 -> full correction
                    # coherence=0 -> no correction
                    #
                    H_amp = (
                        H_amp * C_amp
                    )

                else:

                    H_amp[
                        C_amp < coherence_min
                    ] = 0.0

            # Remove DC explicitly
            H_amp[0] = 0.0

            # -----------------------------------------------------
            # Predicted science pickup in frequency space
            # -----------------------------------------------------
            predicted_fft = (
                ref_fft
                * H_amp[None, :]
            )

            # Back to detector rows
            pickup = np.fft.irfft(
                predicted_fft,
                n=ny,
                axis=1,
            )

            # -----------------------------------------------------
            # Apply to this amplifier
            # -----------------------------------------------------
            a0 = amp * chsize
            a1 = (amp + 1) * chsize

            x0 = a0
            x1 = a1

            if not correct_ref_pixels:

                if amp == 0:
                    x0 = max(
                        x0,
                        nl,
                    )

                if amp == nchans - 1:
                    x1 = min(
                        x1,
                        nx - nr,
                    )

            model_cube[
                :,
                :,
                x0:x1
            ] = pickup[:, :, None]

    else:

        raise ValueError(
            f"Unknown method='{method}'. "
            "Use 'direct' or 'transfer'."
        )

    # -------------------------------------------------------------
    # Apply correction
    # -------------------------------------------------------------
    arr -= model_cube

    # -------------------------------------------------------------
    # Restore dimensionality
    # -------------------------------------------------------------
    if single_frame:

        arr = arr[0]
        model_cube = model_cube[0]

    if return_model:
        fits.writeto('moel.fits',model_cube,overwrite=True)
        return arr

    return arr



def calc_col_smooth(refvals, data_shape, perint=False, edge_wrap=False,
	delt=5.24E-4, savgol=False, winsize=31, order=3, **kwargs):
    """Perform optimal smoothing of side ref pix
    Generates smoothed version of column reference values.
    Smooths values from calc_avg_cols() via FFT.
    Parameters
    ----------
    refvals : ndarray
        Averaged column reference pixels
    data_shape : tuple
        Shape of original data (nz,ny,nx)
    Keyword Arguments
    =================
    perint : bool
        Smooth side reference pixel per int, otherwise per frame.
    edge_wrap : bool
        Add a partial frames to the beginning and end of each averaged
        time series pixels in order to get rid of edge effects.
    delt : float
        Time between reference pixel samples.
    savgol : bool
        Using Savitsky-Golay filter method rather than FFT.
    winsize : int
        Size of the window filter.
    order : int
        Order of the polynomial used to fit the samples.
    """
    #in perint=True, you are smoothing across frames and rows together,
    #like treating the entire cube’s ref signal as one long time-series.
    nz,ny,nx = data_shape
    if perint: # per integration, treats the entire (nz, ny) array as one flattened 1D series
    	if edge_wrap: # Wrap around to avoid edge effects, "mirror" the first and last frames to reduce edge ringing:
            #These are stacked above and below refvals
            #After smoothing, will strip off these mirrored sections and reshape.
    		refvals2 = np.vstack((refvals[0][::-1], refvals, refvals[-1][::-1]))
    		if savgol: # SavGol filter
    			refvals_smoothed2 = savgol_filter(refvals2.ravel(), winsize, order, delta=1)
    		else: # Or "optimal" smoothing algorithm
    			refvals_smoothed2 = smooth_fft(refvals2, delt)
    		refvals_smoothed = refvals_smoothed2[ny:-ny].reshape(refvals.shape)
    	else:
    		if savgol:
    			refvals_smoothed = savgol_filter(refvals.ravel(), winsize, order, delta=1)
    		else:
    			refvals_smoothed = smooth_fft(refvals, delt)
    		refvals_smoothed = refvals_smoothed.reshape(refvals.shape)
    else: #smooth each frame’s row-series separately
    	refvals_smoothed = []
    	if edge_wrap: # Wrap around to avoid edge effects
    		for ref in refvals: #(ny,)
                #mirror to handle FFT more gently on edges
    			ref2 = np.concatenate((ref[:ny//2][::-1], ref, ref[ny//2:][::-1]))
    			if savgol:
    				ref_smth = savgol_filter(ref2, winsize, order, delta=1)
    			else:
    				ref_smth = smooth_fft(ref2, delt)
    			refvals_smoothed.append(ref_smth[ny//2:ny//2+ny])
    		refvals_smoothed = np.array(refvals_smoothed)
    	else:
    		for ref in refvals:
    			if savgol:
    				ref_smth = savgol_filter(ref, winsize, order, delta=1)
    			else:
    				ref_smth = smooth_fft(ref, delt)
    			refvals_smoothed.append(ref_smth)
    		refvals_smoothed = np.array(refvals_smoothed)
    return refvals_smoothed


def smooth_fft(data, delt, first_deriv=False, second_deriv=False):
    """
    Optimal FFT smoothing of evenly sampled data.

    Based on the Kosarev & Pantos filtering approach.
    """

    Dat = np.asarray(data, dtype=float).flatten()
    N = Dat.size

    if N < 8:
        if second_deriv:
            return Dat.copy(), np.zeros_like(Dat), np.zeros_like(Dat)
        elif first_deriv:
            return Dat.copy(), np.zeros_like(Dat)
        else:
            return Dat.copy()

    Pi2 = 2 * np.pi
    OMEGA = Pi2 / (N * delt)
    X = np.arange(N) * delt

    # ------------------------------------------------
    # Center and remove linear baseline
    # ------------------------------------------------
    Dat_m = Dat - np.mean(Dat)
    SLOPE = (Dat_m[-1] - Dat_m[0]) / max(N - 2, 1)
    Dat_b = Dat_m - Dat_m[0] - SLOPE * X / delt

    # ------------------------------------------------
    # FFT and power spectrum
    # ------------------------------------------------
    Dat_F = np.fft.rfft(Dat_b)
    Dat_P = np.abs(Dat_F) ** 2

    nfreq = len(Dat_P)
    max_j = nfreq - 1

    # ------------------------------------------------
    # Estimate white-noise floor from N/4 to N/2
    # ------------------------------------------------
    i1 = max(1, int((N - 1) / 4))
    i2 = min(nfreq, int((N - 1) / 2) + 1)

    if i2 <= i1:
        Noise = np.nanmedian(Dat_P[1:]) if nfreq > 1 else 0.0
    else:
        Noise = np.mean(Dat_P[i1:i2])

    if not np.isfinite(Noise) or Noise <= 0:
        Noise = np.nanmedian(Dat_P[1:]) if nfreq > 1 else 0.0

    if not np.isfinite(Noise) or Noise <= 0:
        Noise = 1e-30

    # ------------------------------------------------
    # Find J0 where signal reaches the noise floor
    # ------------------------------------------------
    J0 = 2
    search_max = min(int(N / 4), max_j - 3)

    for i in range(1, search_max + 1):
        sig0, sig1, sig2, sig3 = Dat_P[i:i + 4]

        if (sig0 < Noise) and ((sig1 < Noise) or (sig2 < Noise) or (sig3 < Noise)):
            J0 = i
            break

    J0 = max(1, min(J0, max_j - 1))

    # ------------------------------------------------
    # Fit straight line to log power spectrum from 1 to J0
    # ------------------------------------------------
    ii = np.arange(1, J0 + 1)
    power = Dat_P[1:J0 + 1]

    power = np.where(power > 0, power, 1e-30)
    logvals = np.log(power)

    XY = np.sum(ii * logvals)
    XX = np.sum(ii ** 2)
    S = np.sum(logvals)

    XM = (2.0 + J0) / 2.0
    YM = S / J0

    denom = XX - J0 * XM * XM

    if np.isfinite(denom) and abs(denom) > 0:
        A1 = (XY - J0 * XM * YM) / denom
    else:
        A1 = -1.0

    B1 = YM - A1 * XM

    # ------------------------------------------------
    # Compute J1 and clamp it to valid rFFT range
    # ------------------------------------------------
    if np.isfinite(A1) and A1 != 0:
        J1 = int(np.ceil((np.log(0.01 * Noise) - B1) / A1))
    else:
        J1 = J0 + 1

    if J1 < J0:
        J1 = J0 + 1

    J1 = min(J1, max_j)

    # ------------------------------------------------
    # Build Kosarev-Pantos filter window
    # ------------------------------------------------
    LOPT = np.zeros_like(Dat_P)

    LOPT[0:J0 + 1] = Dat_P[0:J0 + 1] / (Dat_P[0:J0 + 1] + Noise)

    if J1 > J0:
        i_arr = np.arange(J0 + 1, J1 + 1)

        expo = A1 * i_arr + B1
        expo = np.clip(expo, -700, 700)

        model = np.exp(expo)
        LOPT[J0 + 1:J1 + 1] = model / (model + Noise)

    # ------------------------------------------------
    # Apply filter and optionally compute derivatives
    # ------------------------------------------------
    if second_deriv:
        ndiff = 3
    elif first_deriv:
        ndiff = 2
    else:
        ndiff = 1

    outputs = []

    for diff in range(ndiff):
        Fltr_Spectrum = np.zeros_like(Dat_F, dtype=complex)

        i_start = 1
        n2 = nfreq - 1

        FltrCoef = LOPT[i_start:].astype(np.complex128)

        iW = ((np.arange(n2) + i_start) * OMEGA * 1j) ** diff

        Fltr_Spectrum[i_start:] = Dat_F[i_start:] * FltrCoef * iW

        Fltr_Spectrum[0] = 0 if diff > 0 else Dat_F[0]

        Dat_T = np.fft.irfft(Fltr_Spectrum, n=N)

        if diff == 0:
            outputs.append(np.real(Dat_T) + Dat[0] + SLOPE * X / delt)
        elif diff == 1:
            outputs.append(np.real(Dat_T) + SLOPE / delt)
        elif diff == 2:
            outputs.append(np.real(Dat_T))

    if second_deriv:
        return outputs[0], outputs[1], outputs[2]
    elif first_deriv:
        return outputs[0], outputs[1]
    else:
        return outputs[0]

import numpy as np

from scipy.ndimage import (
    binary_dilation,
    median_filter,
)
from astropy.stats import sigma_clipped_stats


def correct_evolving_row_pickup(
    cube,
    *,
    nleft=4,
    nright=4,
    ntop=4,
    nbot=4,

    # Source / outlier masking
    sigma_thresh=4.0,
    dilate_iter=2,
    min_good_fraction=0.25,

    # Stripe spatial scale
    highpass_size=101,

    # Correction behavior
    remove_full_row_pattern=True,

    # Detector geometry
    nchans=4,
    per_amp=False,

    # Optional iterations
    n_passes=1,

    # Outputs
    return_model=False,
    return_diagnostics=False,
):
    """
    Correct evolving horizontal pickup noise in an HxRG ramp.

    The pickup is estimated from consecutive-read differences:

        diff[r] = cube[r+1] - cube[r]

    For each difference image, a robust science-pixel row profile is
    calculated after masking compact sources / outliers.

    The row profile is then high-pass filtered along detector rows so
    that only stripe-scale structure is removed while broad spatial
    structure is preserved.

    Parameters
    ----------
    cube : ndarray
        Input ramp cube with shape:

            (nread, ny, nx)

    nleft, nright : int
        Number of left/right reference columns excluded from the
        science-pixel row estimator.

    ntop, nbot : int
        Number of top/bottom reference rows excluded from the
        science-pixel row estimator.

    sigma_thresh : float
        Pixels farther than sigma_thresh * robust_std from the
        difference-image median are masked.

        The mask is symmetric, so both positive and negative outliers
        are removed.

    dilate_iter : int
        Number of binary dilation iterations applied to the outlier
        mask. This helps mask PSF/spectral wings around bright sources.

    min_good_fraction : float
        Minimum fraction of unmasked pixels required in a row.

        Rows with fewer valid pixels are interpolated from neighboring
        rows.

    highpass_size : int or None
        Median-filter width along detector rows used to estimate the
        slowly varying component.

        The pickup correction is:

            row_profile - median_filter(row_profile)

        Suggested test values:

            51
            101
            201

        Set to None to subtract the complete row profile.

    remove_full_row_pattern : bool
        Controls what component of the difference row profile is
        removed.

        True
            Remove the full stripe-scale row structure from every read
            difference.

            This is the recommended mode for the current SCALES pickup
            investigation.

        False
            First subtract the median row profile across all read
            differences and remove only the read-variable component.

            This is more conservative but may preserve the pickup
            component that propagates into the final slope.

    nchans : int
        Number of amplifier channels.

    per_amp : bool
        False
            Estimate one row pickup profile using the full science
            region.

        True
            Estimate a separate row pickup profile for every amplifier.

    n_passes : int
        Number of times the correction is repeated.

        Start with:

            n_passes=1

    return_model : bool
        If True, return the cumulative read-level pickup model.

    return_diagnostics : bool
        If True, return diagnostic profiles and masks.

    Returns
    -------
    corrected : ndarray
        Corrected ramp cube.

    model : ndarray, optional
        Cumulative read-level pickup model.

    diagnostics : dict, optional
        Dictionary containing row profiles and intermediate models.
    """

    # =============================================================
    # Input handling
    # =============================================================

    arr = np.asarray(cube)

    if arr.ndim != 3:
        raise ValueError(
            "cube must have shape (nread, ny, nx); "
            f"got {arr.shape}"
        )

    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(
            np.float32,
            copy=True,
        )
    else:
        arr = arr.astype(
            np.float64,
            copy=True,
        )

    nread, ny, nx = arr.shape

    if nread < 2:
        raise ValueError(
            "At least two reads are required."
        )

    if nx % nchans != 0:
        raise ValueError(
            f"Detector width {nx} is not divisible "
            f"by nchans={nchans}."
        )

    if not (0 <= min_good_fraction <= 1):
        raise ValueError(
            "min_good_fraction must be between 0 and 1."
        )

    chsize = nx // nchans

    sci_y0 = nbot
    sci_y1 = ny - ntop

    sci_x0 = nleft
    sci_x1 = nx - nright

    if sci_y1 <= sci_y0:
        raise ValueError(
            "ntop/nbot leave no science rows."
        )

    if sci_x1 <= sci_x0:
        raise ValueError(
            "nleft/nright leave no science columns."
        )

    working = arr.copy()

    total_model = np.zeros_like(
        working,
        dtype=np.float64,
    )

    diagnostics = {}


    # =============================================================
    # Helper: interpolate invalid row values
    # =============================================================

    def fill_bad_rows(profile):
        """
        Replace invalid row estimates by interpolation.
        """

        p = np.asarray(
            profile,
            dtype=float,
        ).copy()

        row_index = np.arange(
            p.size
        )

        good = np.isfinite(p)

        ngood = np.count_nonzero(good)

        if ngood >= 2:

            p[~good] = np.interp(
                row_index[~good],
                row_index[good],
                p[good],
            )

        elif ngood == 1:

            p[:] = p[good][0]

        else:

            p[:] = 0.0

        return p


    # =============================================================
    # Helper: robust row profile from one difference image
    # =============================================================

    def estimate_row_profile(
        diff,
        x0,
        x1,
    ):
        """
        Estimate one robust row profile from a detector sub-region.
        """

        region = diff[
            sci_y0:sci_y1,
            x0:x1,
        ]

        # ---------------------------------------------------------
        # Robust global statistics for outlier/source masking
        # ---------------------------------------------------------

        _, med, std = sigma_clipped_stats(
            region,
            sigma=3.0,
            maxiters=5,
        )

        if (
            not np.isfinite(std)
            or std <= 0
        ):
            std = np.nanstd(
                region
            )

        if (
            not np.isfinite(std)
            or std <= 0
        ):
            std = 1.0

        # ---------------------------------------------------------
        # Mask bright and negative outliers
        # ---------------------------------------------------------

        mask = (
            np.abs(region - med)
            > sigma_thresh * std
        )

        if dilate_iter > 0:

            mask = binary_dilation(
                mask,
                iterations=dilate_iter,
            )

        values = np.where(
            mask,
            np.nan,
            region,
        )

        # ---------------------------------------------------------
        # Robust science-row estimator
        # ---------------------------------------------------------

        row_profile_mid = np.nanmedian(
            values,
            axis=1,
        )

        good_fraction = np.mean(
            np.isfinite(values),
            axis=1,
        )

        row_profile_mid[
            good_fraction < min_good_fraction
        ] = np.nan

        row_profile_mid = fill_bad_rows(
            row_profile_mid
        )

        # ---------------------------------------------------------
        # Put into full detector-row coordinates
        # ---------------------------------------------------------

        row_profile = np.zeros(
            ny,
            dtype=float,
        )

        row_profile[
            sci_y0:sci_y1
        ] = row_profile_mid

        # ---------------------------------------------------------
        # Remove scalar DC level
        # ---------------------------------------------------------

        dc = np.nanmedian(
            row_profile[
                sci_y0:sci_y1
            ]
        )

        row_profile[
            sci_y0:sci_y1
        ] -= dc

        return (
            row_profile,
            mask,
            good_fraction,
        )


    # =============================================================
    # Correction passes
    # =============================================================

    for pass_index in range(n_passes):

        # ---------------------------------------------------------
        # Consecutive-read differences
        # ---------------------------------------------------------

        diffs = (
            working[1:]
            - working[:-1]
        )

        ndiff = diffs.shape[0]

        masks_all = []


        # =========================================================
        # Measure row profile for every difference
        # =========================================================

        if per_amp:

            profiles = np.zeros(
                (
                    ndiff,
                    nchans,
                    ny,
                ),
                dtype=float,
            )

            good_fraction_all = np.zeros(
                (
                    ndiff,
                    nchans,
                    sci_y1 - sci_y0,
                ),
                dtype=float,
            )

            for r in range(ndiff):

                masks_read = []

                for amp in range(nchans):

                    amp_x0 = (
                        amp * chsize
                    )

                    amp_x1 = (
                        (amp + 1)
                        * chsize
                    )

                    x0 = max(
                        amp_x0,
                        sci_x0,
                    )

                    x1 = min(
                        amp_x1,
                        sci_x1,
                    )

                    if x1 <= x0:
                        continue

                    (
                        profile,
                        mask,
                        good_fraction,
                    ) = estimate_row_profile(
                        diffs[r],
                        x0,
                        x1,
                    )

                    profiles[
                        r,
                        amp,
                    ] = profile

                    good_fraction_all[
                        r,
                        amp,
                    ] = good_fraction

                    masks_read.append(
                        mask
                    )

                masks_all.append(
                    masks_read
                )

        else:

            profiles = np.zeros(
                (
                    ndiff,
                    ny,
                ),
                dtype=float,
            )

            good_fraction_all = np.zeros(
                (
                    ndiff,
                    sci_y1 - sci_y0,
                ),
                dtype=float,
            )

            for r in range(ndiff):

                (
                    profile,
                    mask,
                    good_fraction,
                ) = estimate_row_profile(
                    diffs[r],
                    sci_x0,
                    sci_x1,
                )

                profiles[r] = profile

                good_fraction_all[
                    r
                ] = good_fraction

                masks_all.append(
                    mask
                )


        # =========================================================
        # Decide which row component to remove
        # =========================================================

        static_profile = np.nanmedian(
            profiles,
            axis=0,
        )

        if remove_full_row_pattern:

            correction_profiles = (
                profiles.copy()
            )

        else:

            correction_profiles = (
                profiles
                - static_profile[
                    None,
                    ...
                ]
            )


        # =========================================================
        # Keep stripe-scale structure only
        # =========================================================

        if (
            highpass_size is not None
            and highpass_size > 1
        ):

            if per_amp:

                for r in range(ndiff):

                    for amp in range(
                        nchans
                    ):

                        p = (
                            correction_profiles[
                                r,
                                amp,
                            ]
                        )

                        slow = median_filter(
                            p,
                            size=highpass_size,
                            mode="nearest",
                        )

                        correction_profiles[
                            r,
                            amp,
                        ] = (
                            p - slow
                        )

            else:

                for r in range(ndiff):

                    p = (
                        correction_profiles[
                            r
                        ]
                    )

                    slow = median_filter(
                        p,
                        size=highpass_size,
                        mode="nearest",
                    )

                    correction_profiles[
                        r
                    ] = (
                        p - slow
                    )


        # =========================================================
        # Build difference-space pickup model
        # =========================================================

        diff_model = np.zeros_like(
            diffs,
            dtype=np.float64,
        )

        if per_amp:

            for r in range(ndiff):

                for amp in range(
                    nchans
                ):

                    amp_x0 = (
                        amp * chsize
                    )

                    amp_x1 = (
                        (amp + 1)
                        * chsize
                    )

                    x0 = max(
                        amp_x0,
                        sci_x0,
                    )

                    x1 = min(
                        amp_x1,
                        sci_x1,
                    )

                    if x1 <= x0:
                        continue

                    diff_model[
                        r,
                        :,
                        x0:x1,
                    ] = (
                        correction_profiles[
                            r,
                            amp,
                            :,
                        ][:, None]
                    )

        else:

            diff_model[
                :,
                :,
                sci_x0:sci_x1,
            ] = (
                correction_profiles[
                    :,
                    :,
                    None,
                ]
            )


        # =========================================================
        # Integrate difference correction back into read space
        # =========================================================

        read_model = np.zeros_like(
            working,
            dtype=np.float64,
        )

        # First read is kept unchanged.
        #
        # If
        #
        # diff_model[0] = correction for read1-read0
        #
        # then
        #
        # read_model[1] = diff_model[0]
        # read_model[2] = diff_model[0] + diff_model[1]
        # ...

        read_model[1:] = np.cumsum(
            diff_model,
            axis=0,
        )


        # =========================================================
        # Apply correction
        # =========================================================

        working -= read_model

        total_model += read_model


        # =========================================================
        # Store diagnostics
        # =========================================================

        diagnostics[
            f"pass_{pass_index}"
        ] = {
            "diffs": diffs.copy(),

            "row_profiles":
                profiles.copy(),

            "static_profile":
                static_profile.copy(),

            "correction_profiles":
                correction_profiles.copy(),

            "diff_model":
                diff_model.copy(),

            "read_model":
                read_model.copy(),

            "good_fraction":
                good_fraction_all.copy(),

            "masks":
                masks_all,
        }


    # =============================================================
    # Return
    # =============================================================

    corrected = working

    outputs = [
        corrected
    ]

    if return_model:
        outputs.append(
            total_model
        )

    if return_diagnostics:
        outputs.append(
            diagnostics
        )

    if len(outputs) == 1:
        return outputs[0]

    return tuple(outputs)
