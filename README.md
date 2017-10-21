# linalgcpp
A linear algebra library

# Requirements
* BLAS - for dense matrix operation
* LAPACK - for solving systems of equations, SVD, eigenproblems, etc.
* Tested with gcc-7.1.1, c++14

# Build
* Ensure requirements listed above are available on your system. 
* Copy ```Makefile.in``` to ```Makefile```
* Modify the make file if the required libraries are not in the standard locations.
* ```make```
