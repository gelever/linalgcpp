# DenseMatrix Examples

Provides example code for `DenseMatrix`.

# Basics
`dense_basics.cpp` covers the basics including:
* Constructors
* Parsing
* Printing
* Matrix size
* Access Operations
* Matrix-Vector multiplication
* Matrix-Matrix multiplication

# LU Decomposition
`dense_lu.cpp` shows how one would implement a textbook version of the [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition).
* LU decomposition
* Solve linear system `Ax=b`:
  * A = LU
  * Forward elimination for `Ly=b`
  * Back solve for `Ux=y`
* Compare solve times as matrix size grows

# QR Decomposition
`dense_qr.cpp` shows how one would implement a textbook version of the [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition).
 * QR decomposition
 * Solve least squares problem `Ax=b`:
   * A = QR
   * Back solve `Rx=Q^T b`
 * Compute eigen decomposition `A=Q T Q^T` using QR
 * Compare solve times as matrix size grows
