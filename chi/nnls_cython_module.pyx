# nnls_cython_module.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np # Import C-level NumPy API
from scipy.optimize import lsq_linear
import scipy.sparse as sp

from cython.parallel import prange # 
# Import the `prange` function, which is the Cython equivalent of `range` 
#for parallel loops using OpenMP.

import time

# Define C type for float64
ctypedef np.float64_t DOUBLE_t

# --------------------------------------------------------------------------
# Function to solve a single NNLS problem
# --------------------------------------------------------------------------
# CORRECTED ARGUMENT ORDER: non-defaults first
def solve_nnls_flux_cy(object R_matrix,
                       np.ndarray[DOUBLE_t, ndim=1] data_vector,
                       dict lsq_options, # Moved before default arg
                       int task_id=-1, # Default arg last
                       ):
    """
    Cython version (wrapper) for solving NNLS for a single data vector.
    Intended to be called from a prange loop. Releases GIL via lsq_linear.
    Args:
        R_matrix: The sparse rectification matrix (passed as object).
        data_vector: The 1D flattened data vector for this task.
        lsq_options: Dictionary of options for lsq_linear.
        task_id: Identifier for the task (for logging). Defaukt is -1.
    Returns:
        1D NumPy array containing the solution vector A.
    """
    start_time = time.time()
    current_options = {'verbose': 0} #Initialize options for lsq_linear with a default.

    if lsq_options:
        current_options.update(lsq_options)
    #update the default option with the user provided input.

    # --- Call the SciPy function (Assumed to release GIL internally) ---
    res = lsq_linear(R_matrix, data_vector, bounds=(0, np.inf), **current_options)
    # Call the SciPy least-squares solver
    # - `R_matrix`, `data_vector`: Input matrix and vector.
    # - `bounds=(0, np.inf)`: Enforces the non-negativity constraint (NNLS).
    # - `**current_options`: Passes solver options like tolerance, max iterations.
    # - **CRITICAL ASSUMPTION:** This call is expected to release the GIL internally 
    #for the bulk of its computation, allowing other threads in the `prange` loop to 
    #compute simultaneously.

    # -------------------------------------------------------------------
    end_time = time.time()

    return res.x

# --------------------------------------------------------------------------
# Function containing the parallel loop
# --------------------------------------------------------------------------
cpdef np.ndarray[DOUBLE_t, ndim=2] run_parallel_nnls_cython(
                                        object R_matrix,
                                        list data_vectors_list,
                                        dict nnls_options,
                                        int num_threads=-1
                                        ):
    """
    Runs solve_nnls_flux_cy in parallel using cython.parallel.prange.
    cpdef` makes this function callable efficiently from both Python and C (other Cython code).
    R_matrix:  SciPy sparse matrix
    data_vectors_list : A standard Python list containing the 1D NumPy arrays
    - nnls_options : Dictionary of options passed down to `solve_nnls_flux_cy`
    - num_threads : Desired number of OpenMP threads (-1 or 0 for default)

    """

    cdef int num_tasks = len(data_vectors_list) #get tottal number of tasks
    cdef int n_fluxes = 0
    cdef int i
    # We need these variables declared outside the gil block if used inside/outside
    cdef np.ndarray[DOUBLE_t, ndim=1] current_data_vector
    cdef np.ndarray[DOUBLE_t, ndim=1] result_vector

    if num_tasks == 0:
        print("Cython Warning: Empty data_vectors_list provided.")

        return np.empty((0, 0), dtype=np.float64)



    try:
        n_fluxes = R_matrix.shape[1]
    except (AttributeError, IndexError):
        print("Cython Warning: Could not determine n_fluxes from R_matrix shape. Running one task serially.")
        try:
            # Call with corrected argument order
            temp_result = solve_nnls_flux_cy(R_matrix, data_vectors_list[0], nnls_options, 0)
            n_fluxes = temp_result.shape[0]
            # Get the length of the solution vector from this single run.

        except Exception as e:
            print(f"Cython Error: Failed to run even one task to determine n_fluxes: {e}")
            raise ValueError("Could not determine the number of fluxes (output dimension).") from e



    if n_fluxes <= 0:
         raise ValueError(f"Determined number of fluxes ({n_fluxes}) is invalid.")
         #ensure the determined nuber of fluxes are valid.

    print(f"Cython: Preparing parallel run for {num_tasks} tasks with {n_fluxes} fluxes each.")
    cdef np.ndarray[DOUBLE_t, ndim=2] A_optimal_array = np.empty((num_tasks, n_fluxes), dtype=np.float64)
    #define the output amplitude array

    cdef int effective_num_threads # Declare C int variable

    if num_threads <= 0:
        effective_num_threads = 0 
        # 0 implies prange/OpenMP to use the default number of threads (OMP_NUM_THREADS or max cores). 
        print_num_threads = "default"
    else:
        effective_num_threads = num_threads #user specified number
        print_num_threads = str(effective_num_threads)

    print(f"Cython: Running prange with num_threads={print_num_threads}...")
    start_time_prange = time.time()

    # prange handles the parallel region setup
    # nogil=True releases GIL for the loop *structure* and allows C-level parallelism
    for i in prange(num_tasks, nogil=True, schedule='dynamic', num_threads=effective_num_threads):
        # This is the parallel loop
        # Iterates from 0 to num_tasks-1
        # nogil=True : Releases the Python GIL. Threads can now run C code concurrently.
        # schedule='dynamic' : Work is divided among threads in fixed chunks initially. Good if tasks have unequal duration.
        # num_threads : Specifies how many OpenMP threads to use for this loop

        with gil:
            # Access Python list (requires GIL)
            current_data_vector = <np.ndarray>data_vectors_list[i]

            result_vector = solve_nnls_flux_cy(R_matrix, current_data_vector, nnls_options, i)
            # Assign result to NumPy array slice (requires GIL)
            A_optimal_array[i, :] = result_vector


    end_time_prange = time.time()
    print(f"Cython: prange loop finished in {end_time_prange - start_time_prange:.2f} seconds (requested {print_num_threads} threads).")

    return A_optimal_array