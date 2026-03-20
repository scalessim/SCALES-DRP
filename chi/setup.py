from setuptools import setup, Extension 
from Cython.Build import cythonize #to process .pyx file
import numpy
import sys
import os
import platform

# Define compiler/linker flags for OpenMP &Initialize
compile_args = []
link_args = []

# --- More Specific Platform and Compiler Checks ---
if sys.platform == 'darwin': # Check specifically for macOS
    print("Detected macOS platform.")
    libomp_prefix_arm = '/opt/homebrew/opt/libomp' #for Apple Silicon
    libomp_prefix_x86 = '/usr/local/opt/libomp'    #for Intel Macs
    libomp_prefix = None

    if platform.machine() == 'arm64' and os.path.isdir(libomp_prefix_arm):
         libomp_prefix = libomp_prefix_arm
    elif os.path.isdir(libomp_prefix_x86): # Check x86 path as fallback or if on x86
         libomp_prefix = libomp_prefix_x86

    if libomp_prefix:
        print(f"Found libomp installation at: {libomp_prefix}")
        compile_args = [
            '-Xclang', '-fopenmp', # Pass -fopenmp to the clang frontend
            f'-I{libomp_prefix}/include' # Include path for omp.h
        ]
        link_args = [
            f'-L{libomp_prefix}/lib', # Library path for libomp
            '-lomp' # Link against libomp
        ]
        print(f"Using macOS clang OpenMP flags: {compile_args} {link_args}")
    else:
        print("*"*40)
        print("WARNING: libomp installation not found in standard Homebrew locations.")
        print("         Please install using 'brew install libomp'.")
        print("         OpenMP support will likely be disabled.")
        print("*"*40)

elif sys.platform.startswith('win'):
    # Windows (MSVC compiler )
    print("Detected Windows platform.")
    compile_args = ['/openmp']
    link_args = []
    print("Attempting to enable OpenMP for MSVC compiler (/openmp)")

elif sys.platform.startswith('linux'):
    # Linux (usually GCC)
    print("Detected Linux platform.")
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']
    print("Attempting to enable OpenMP for GCC-like compiler (-fopenmp)")

else:
    # Fallback for other systems
    print(f"Detected platform: {sys.platform}. Attempting generic -fopenmp flags.")
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']

#Define extension module

extensions = [
    Extension(
        "nnls_cython_module",             # Name of the module to import in Python
        ["nnls_cython_module.pyx"],       # List of source files
        include_dirs=[numpy.get_include()], # Include NumPy headers
        extra_compile_args=compile_args,  # Add correct OpenMP compile flag
        extra_link_args=link_args,        # Add correct OpenMP link flag
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] 
    )
]

print("\n--- DEBUG: About to call setup() ---") 

try:
    #main setup function call
    setup(
        name="NNLS Cython Module", #Name of the package
        version="0.1",
        description="Parallel NNLS solver using Cython and OpenMP",
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                'language_level': "3",
                },
            force=True # Force Cython to re-compile .pyx to .c
            )
    )
    print("--- DEBUG: setup() call finished ---") 
except Exception as e:
    print(f"--- DEBUG: setup() FAILED with exception: {e} ---") 
    import traceback
    traceback.print_exc()

print("-" * 30)
print("Build complete message (if setup didn't exit).")
print("-" * 30)


