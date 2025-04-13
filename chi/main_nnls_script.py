import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz
from astropy.io import fits
import os
import sys 

########################################################################
#Thread control

num_outer_threads = '8' # <--- Set desired number for your main loop (prange)
os.environ['OMP_NUM_THREADS'] = num_outer_threads
# Set the environment variable that OpenMP (used by prange) reads to determine the default number of threads.

# Explicitly set the number of threads for the INNER libraries to 1
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1' # For macOS Accelerate
os.environ['NUMEXPR_NUM_THREADS'] = '1'

#Setting these to '1' prevents libraries called *within* the parallel loop
#This avoids thread oversubscription, which kills performance.


print(f"--- Thread Control ---")
print(f"Set OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')} (for outer prange loop)")
print(f"Set MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS')} (for inner BLAS/LAPACK)")
print(f"Set OPENBLAS_NUM_THREADS = {os.environ.get('OPENBLAS_NUM_THREADS')} (for inner BLAS/LAPACK)")
print(f"Set VECLIB_MAXIMUM_THREADS = {os.environ.get('VECLIB_MAXIMUM_THREADS')} (for inner Accelerate)")
print(f"Set NUMEXPR_NUM_THREADS = {os.environ.get('NUMEXPR_NUM_THREADS')} (for inner Numexpr)")
print(f"--- Starting Main Script ---")

####################################################################################

# --- Attempt to import the compiled Cython module ---
try:
    # Ensure the directory containing the compiled .so/.pyd file is in the Python path
    # If setup.py build_ext --inplace was used, the .so/.pyd file in the current directory.
    import nnls_cython_module
    print("Successfully imported compiled Cython module: nnls_cython_module")
    USE_CYTHON = True
except ImportError as e:
    #If the import fails  =====> module not compiled or not found
    print("="*50)
    print(f"ERROR: Could not import compiled Cython module 'nnls_cython_module'.")
    print(f"ImportError: {e}")
    print("Please ensure the module has been compiled successfully:")
    print("1. You need a C compiler (GCC, Clang, or MSVC).")
    print("2. You need Cython installed (`pip install cython`).")
    print("3. You need NumPy installed (`pip install numpy`).")
    print("4. Run: python setup.py build_ext --inplace")
    print("5. Ensure OpenMP runtime library is installed if setup enabled it.")
    print("Falling back to sequential execution for demonstration.")
    print("="*50)
    USE_CYTHON = False # Flag indicate Cython is not available

    # Define a dummy sequential function if Cython fails, for basic testing
    def run_sequential_nnls(R_matrix, data_vectors_list, nnls_options):
        from scipy.optimize import lsq_linear # Import only if needed for fallback
        results = []
        for i, dv in enumerate(data_vectors_list):
             print(f"Running task {i} sequentially...")
             opts = {'verbose': 0}; opts.update(nnls_options)
             res = lsq_linear(R_matrix, dv, bounds=(0, np.inf), **opts)
             results.append(res.x)
        if results:
            return np.array(results) # Stack into 2D array
        else:
            return np.empty((0,0), dtype=np.float64)

######################################################################################
# --- Configuration & Paths ---


# Define IMG_DIM based on your slicing (assuming 512x512 tiles)
IMG_DIM = 512
N_TILES_SIDE = 4 # Assuming 2048 / 512 = 4

# --- File Paths ---

try:
    # Base directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback if __file__ is not defined (e.g., in interactive session)
    script_dir = os.getcwd()

data_path = os.path.join('/Users/athira/Desktop/scales_drp/data/new_cal_data/') # Example path structure
matrix_path = os.path.join('/Users/athira/Desktop/scales_drp/chi_extraction/trim_512/') # Example path structure

data_fits_file = 'A_star_nosky_noshot_nodet.fits'
matrix_file = "512_rectmat_1.npz"

full_data_path = os.path.join(data_path, data_fits_file)
full_matrix_path = os.path.join(matrix_path, matrix_file)

###############################################################################

# --- Main Execution ---
if __name__ == "__main__":

    print(f"\n--- Loading Data: {full_data_path} ---")
    data_image = fits.getdata(full_data_path)[0]

    # --- Data Preparation ---
    print("\n--- Data Preparation ---")
    data_vectors = []
    expected_tasks = N_TILES_SIDE * N_TILES_SIDE
    print(f"Splitting into {N_TILES_SIDE}x{N_TILES_SIDE} = {expected_tasks} tiles of size {IMG_DIM}x{IMG_DIM}...")

    for i in range(N_TILES_SIDE): # Rows of tiles
        for j in range(N_TILES_SIDE): # Columns of tiles
            row_start, row_end = i * IMG_DIM, (i + 1) * IMG_DIM
            col_start, col_end = j * IMG_DIM, (j + 1) * IMG_DIM
            tile = data_image[row_start:row_end, col_start:col_end]
            data_vectors.append(tile.flatten().astype(np.float64))
            #Flatten from 2D to 1D 

    num_tasks = len(data_vectors)

    print(f"Prepared {num_tasks} data vectors.")
#############################################################################

    # --- Load R Matrix ---
    print(f"\n--- Loading R Matrix: {full_matrix_path} ---")

    R_matrix = load_npz(full_matrix_path)
    print(f"R matrix shape: {R_matrix.shape}, Type: {type(R_matrix)}, dtype: {R_matrix.dtype}")
    # Check compatibility
    expected_pixels = IMG_DIM * IMG_DIM

    n_fluxes_expected = R_matrix.shape[1]

####################################################################################

    # --- Parallel NNLS Execution ---
    print("\n--- NNLS Execution ---")
    nnls_options = {'tol': 1e-8, 'max_iter': 500, 'verbose': 0} # Options for lsq_linear inside Cython

    A_optimal_array = None # Initialize result array
    total_nnls_time = 0

    if USE_CYTHON:
        # Determine number of threads (cores) to use for Cython/OpenMP
        num_threads_to_use = int(os.environ.get('OMP_NUM_THREADS', 0)) #os.cpu_count() # Use all logical cores
        # num_threads_to_use = 4 
        print(f"Attempting Cython parallel NNLS solution using up to {num_threads_to_use} threads...")
        start_nnls_time = time.time()

        try:
            # Call the compiled Cython function
            A_optimal_array = nnls_cython_module.run_parallel_nnls_cython(
                R_matrix,           # The sparse matrix
                data_vectors,       # The list of 1D NumPy arrays
                nnls_options,       # Dictionary of options for lsq_linear
                num_threads=num_threads_to_use # Requested number of threads
            )
            # A_optimal_array is now a 2D NumPy array (num_tasks, n_fluxes)
            if A_optimal_array is not None:
                 print(f"Cython function returned result array with shape: {A_optimal_array.shape}")
                 if A_optimal_array.shape != (num_tasks, n_fluxes_expected):
                      print("Warning: Result array shape mismatch!")

        except Exception as e:
             print(f"ERROR during Cython execution: {e}")
             print("Please check the Cython code and compilation process.")
             # Decide how to proceed: exit, or try sequential fallback?
             USE_CYTHON = False # Mark Cython as failed

        end_nnls_time = time.time()
        if A_optimal_array is not None:
            total_nnls_time = end_nnls_time - start_nnls_time
            print(f"Cython parallel NNLS solution finished in {total_nnls_time:.2f} seconds.")
    else:
        print("Cython module not available...")

    # --- Post-processing NNLS Results ---
    print("\n--- Post-processing NNLS Results ---")
    initial_flux_3d = None
    if A_optimal_array is not None and A_optimal_array.size > 0:
        # Results are in the 2D array A_optimal_array[task_index, flux_index]
        print(f"Shape of optimal flux array (tasks, fluxes): {A_optimal_array.shape}")

        print("Concatenating/Flattening results for reshaping...")
        try:
            # Flatten the (num_tasks, n_fluxes) array to a single 1D vector
            combined_A = A_optimal_array.flatten()

            # --- Reshape the combined flux vector ---
            reshape_dims = (54, 108, 108)
            expected_total_elements = np.prod(reshape_dims)
            actual_total_elements = combined_A.size

            print(f"Total flux elements calculated: {actual_total_elements}")
            print(f"Expected elements for reshape {reshape_dims}: {expected_total_elements}")

            if actual_total_elements == expected_total_elements:
                print(f"Reshaping combined flux vector to {reshape_dims}...")
                initial_flux_3d = combined_A.reshape(reshape_dims)
                print(f"Shape of reshaped 3D flux: {initial_flux_3d.shape}")
            else:
                print(f"ERROR: Combined flux vector size ({actual_total_elements}) does not match "
                      f"expected reshape size ({expected_total_elements}). Cannot reshape.")
                initial_flux_3d = None

        except ValueError as e:
            print(f"Error flattening or reshaping results: {e}")
            initial_flux_3d = None
        except Exception as e:
            print(f"An unexpected error occurred during result processing: {e}")
            initial_flux_3d = None
    else:
        print("Skipping post-processing as NNLS results are missing or empty.")


    # --- Calculate Model Images ---
    # This part recalculates the model image for each tile using the solved fluxes.
    # It's done sequentially here. Could potentially be parallelized too if needed.

    print("\n--- Model Image Calculation (Sequential) ---")
    model_images = [None] * num_tasks 
    model_vectors_list = [None] * num_tasks
    total_model_time = 0

    if A_optimal_array is not None and A_optimal_array.shape[0] == num_tasks:
        start_model_time = time.time()
        for i in range(num_tasks):
            try:
                # Use matrix-vector multiplication with the i-th result vector from the 2D array
                model_vector = R_matrix @ A_optimal_array[i] # Shape (N_pixels,)
                model_vectors_list[i] = model_vector
                # Reshape the model vector back to 2D 
                model_images[i] = model_vector.reshape(IMG_DIM, IMG_DIM)
            except MemoryError:
                print(f"ERROR: Not enough memory to calculate R @ A for segment {i}.")
                # Keep None in the list to indicate failure
            except Exception as e:
                print(f"ERROR calculating model image segment {i}: {e}")
                # Keep None

        end_model_time = time.time()
        total_model_time = end_model_time - start_model_time
        print(f"Sequential model calculation finished in {total_model_time:.2f} seconds.")
    else:
         print("Skipping model calculation as NNLS results are missing or incomplete.")


    # --- Reconstruct Full Images & Save ---
    print("\n--- Image Reconstruction & Saving ---")
    full_model_image = None
    full_residual_image = None

    # Check if all model images were calculated successfully
    if None not in model_images:
        print("Reconstructing full model image...")
        # Pre-allocate the full image array
        full_image_shape = (IMG_DIM * N_TILES_SIDE, IMG_DIM * N_TILES_SIDE)
        full_model_image = np.zeros(full_image_shape, dtype=np.float64)
        idx = 0
        for r in range(N_TILES_SIDE): # Tile row
            for c in range(N_TILES_SIDE): # Tile column
                row_start, row_end = r * IMG_DIM, (r + 1) * IMG_DIM
                col_start, col_end = c * IMG_DIM, (c + 1) * IMG_DIM
                full_model_image[row_start:row_end, col_start:col_end] = model_images[idx]
                idx += 1

        print("Calculating full residual image...")

        full_residual_image = data_image.astype(np.float64) - full_model_image

        print(f"Full model image shape: {full_model_image.shape}")
        print(f"Full residual image shape: {full_residual_image.shape}")

        # --- Save Output FITS Files ---
        output_dir = os.path.join(script_dir, "output") # Example output directory
        os.makedirs(output_dir, exist_ok=True)

        model_filename = os.path.join(output_dir, "reconstructed_model_image_cython.fits")
        residual_filename = os.path.join(output_dir, "reconstructed_residual_image_cython.fits")
        flux_filename = os.path.join(output_dir, "initial_flux_3d_cython.fits") # Save 3D flux if created

        #saving the model image
        print(f"Saving full model image to: {model_filename}")
        hdu_model = fits.PrimaryHDU(full_model_image)
        hdu_model.writeto(model_filename, overwrite=True)

        #saving residual image
        print(f"Saving full residual image to: {residual_filename}")
        hdu_resid = fits.PrimaryHDU(full_residual_image)
        hdu_resid.writeto(residual_filename, overwrite=True)

        #save the 3D flux cube if it was successfully created
        if initial_flux_3d is not None:
             print(f"Saving reshaped 3D flux to: {flux_filename}")
             hdu_flux = fits.PrimaryHDU(initial_flux_3d)
             hdu_flux.writeto(flux_filename, overwrite=True)
        else:
             print("Skipping save for 3D flux (not created or reshape failed).")

    else:
        print("Skipping reconstruction and saving as some model image segments failed.")

    print("\n--- Script Finished ---")