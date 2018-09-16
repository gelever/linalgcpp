# Parallel Linear Algebra Library [![Build Status](https://travis-ci.org/gelever/parlinalgcpp.svg?branch=master)](https://travis-ci.org/gelever/parlinalgcpp)
Provides a convenient interface to Hypre parallel solvers.
Based heavily on MFEM's linear algebra interface.

Uses linalgcpp as the serial linear algebra base.

## Requirements
* linalgcpp
* Hypre version 2.10.0b
* c++11

## Format
parlinalgcpp provides several parallel extensions to linalgcpp.
Information found only on the current processor is called local.
This is in contrast to global: information across all processors.

### ParMatrix
ParMatrix is a distributed sparse csr matrix.
Each processor owns a continuous, non-empty, subset
of the global rows and columns. The data is distributed
by rows. Each row contains two non-intersecting blocks.
Local entries whose column index is owned by the processor are placed
in the diagonal block. All other entries are considered off diagonal.

The row and column processor partitions are tracked by arrays of size 3.
The first entry is the global index of the first row in the block
and the second entry is one past the global index of last row in the block.
The third entry contains the total global number of rows or columns.

Consider the following (4, 4) global matrix:
```
|1 2 0 3|
|0 4 5 0|
|0 0 6 7|
|8 0 0 9|
```
and the global matrix partitioned between 2 processors:
```
              diag   offd   row start    col start
Processor 0: |1 2 | |0 3|    [0, 2]       [0, 2]
Processor 0: |0 4 | |5 0|
             ------ -----    ---------    ----------
             offd    diag    row start    col start
Processor 1: |0 0 | |6 7|    [2, 4]       [2, 4]
Processor 1: |8 0 | |0 9|
```

Entries in the off-diagonal block are compressed even further.
Only nonzero columns are kept. A map is kept from the local column
entries to the corresponding global entry.

Processor 0's off diagonal data would be:
```
offd
|0 3|
|5 0|

col map: [2, 3]
```
While Processor 1's off diagonal data would be:
```
offd
|0|
|8|

col map: [0]
```

### ParVector
ParVector is a distributed vector.
Each processor owns a continuous, non-empty, subset
of the global vector.

The processor partition is tracked by an array of size 3.
The first entry is the global index of the start of the block
and the second entry is one past the global index of the end in the block.
The third entry contains the total global size of the vector.

Consider the following global vector of size 4:
```
|1|
|2|
|3|
|4|
```
and the global vector partitioned between 2 processors:
```
             vector  partition
Processor 0: |1|     [0, 2]
Processor 0: |2|
             ---     ------
             vector  partition
Processor 1: |3|     [2, 4]
Processor 1: |4|
```
