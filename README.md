# linalgcpp [![Build Status](https://travis-ci.org/gelever/linalgcpp.svg?branch=master)](https://travis-ci.org/gelever/linalgcpp)
A linear algebra library based heavily on [MFEM](https://github.com/mfem/mfem)'s linear algebra.

# Requirements
* BLAS - for dense matrix operation
* LAPACK - for solving systems of equations, SVD, eigenproblems, etc.
* Tested with gcc-7.1.1, c++11

# Build
* Ensure requirements listed above are available on your system. 
* Standard CMake procedure: `mkdir build && cd build && cmake ..`

# Features
* Sparse Matrix in CSR and Coordinate formats
* Column major Dense Matrix
* Vectors and Vector Views
* Block Matrix and Block Vector
* Abstract Operator
* Basic iterative solvers
* Parser for various formats

# Modules
Several optional modules are available to extend functionality:
* [`parlinalgcpp`](modules/parlinalgcpp) - Distributed linear algebra and solvers using [Hypre](https://github.com/LLNL/hypre)
* [`sparsesolve`](modules/sparsesolver) - Sparse matrix solver using [SuiteSparse/UMFPACK](http://faculty.cse.tamu.edu/davis/suitesparse.html)
* [`partition`](modules/partition) - Graph partitioner using [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)

# Project Goal
The goal of this libary is to provide a safe and convenient linear algebra framework.
Type and memory safety are important to provide confidence with implementing new ideas.
If an error can be caught at compile time, then the library should be able to catch it.
If it compiles, then the library should behave as expected.  There should be no possible
segfaults, memory leaks, invalid reads/writes, etc. from the library.

# Formats
There are many basic building blocks in this library. These provide the 
basic framework to build much more complex applications.
## Matrix
The following matrix will be used to show how data in laid out:
```
|1 2 0 0 |
|0 3 0 0 |
|4 0 0 5 |
```
### SparseMatrix
The sparse matrix is in CSR (Compressed Sparse Row) format.  There are 3 arrays that 
keep track of the data.  Only non-zeros (or explicit zeros) are tracked.
For each entry, the column index and the value are kept in ```indices``` and ```data``` respectively.
The ```indptr``` points to where the row starts 
in both ```indices``` and ```data```.
The following graphic ties everything together:
```
indptr data         indices
[0,    |1 2 - - |   [0, 1,
 2,    |- 3 - - |    1,
 3,    |4 - - 5 |    0, 3]
 5]    
```
### CooMatrix
The coordinate matrix is simply a list of coordinates and the corresponding value.
This requires more space to hold than the SparseMatrix, but is easier to deal with.
Use a CooMatrix when you do not know the data layout beforehand. Then, when all the 
data values have been added, convert to the more effecient SparseMatrix.
Data is kept as a list of tuples:
```
[(0, 0, 1),
 (0, 1, 2),
 (1, 1, 3),
 (2, 0, 4),
 (2, 3, 5)]
```
### LilMatrix
The linked list matrix is an array of linked lists, where each row gets its
own list of coordinate.  This is useful when you need to modify elements
in a coordinate matrix, but searching through the entire list of coordinates would be ineffecient.
When all the data values have been added, convert to the more effecient SparseMatrix.
Each row contains the column index and data value for each entry:
```
[[(0, 1), (1, 2)],
 [(1, 3)],
 [(0, 4), (3, 5)]]
```
### DenseMatrix
The dense matrix keeps track of all entries in the matrix.  It may consume a lot of memory
if the matrix is large.  Consider using a SparseMatrix if most entries are zero.
This matrix is column major.  This means that data is contigious by column.  Keep this
in mind when iterating over the entries, as accessing column by column is much more cache friendly.
Notice the data is kept transposed to the normal mathimatical matrix:
```
[1, 0, 4,
 2, 3, 0,
 0, 0, 0,
 0, 0, 5]
```
### BlockMatrix
The block matrix is a matrix split up into individual CSR matrices.  The example matrix
split up into 4 blocks would be:
```
|1 2 | |0 0 |
|0 3 | |0 0 |
------ ------
|4 0 | |0 5 |
```
## Vectors
The following vector will be used to show how data in layed out:
```
|1|
|2|
|3|
```
### VectorView
A vector view is a non-owning slice of data and the corresponding size.
For example, a view of the last two elements of the example vector would be:
```
vector    VectorView
|1|
|2|<------|2|
|3|<------|3|
```
### Vector
A vector is the same thing as its corresponding mathmatical object.
It is a contigious array of data that knows its size:
```
|1|
|2|
|3|
```
### BlockVector
A block vector is a vector that also knows how it is partitioned.
The data is still contigious.
For example, if the example vector has two blocks:
```
|1|   Block 0
---
|2|   Block 1
|3|
```
## Operator
An operator is anything that has a known size and can apply its action to a vector.
This is useful in many algorithms where only the action is required.  For example,
in Conjugate Gradient, the algorithm itself does not care how the operator multiplies a vector,
it just needs the action.  All the matrices described above are operators.
### BlockOperator
A block operator is an operator where each individual block is a seperate operator.
For example, a block operator used as a preconditioner could have a sparse matrix
in one block, and a sparse solver in another.
## Parser
The parser can read and write the following formats:
* Vector in either ascii or binary format
* SparseMatrix in either ascii or binary format
* Various adjacency lists
* Coordinate lists
* Boolean table
