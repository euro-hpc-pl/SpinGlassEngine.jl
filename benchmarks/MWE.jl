using CUDA
using SparseArrays

dense32 = CUDA.rand(1, 1)
sparse32csc = cu(sprand(Float32, 1, 1, 1.))

dense64 = CUDA.rand(Float64, 1, 1)
sparse64csc = CUSPARSE.CuSparseMatrixCSC{Float64, Int32}(sparse32csc.colPtr, sparse32csc.rowVal, sparse32csc.nzVal, (1, 1))

sparse64csr = CUSPARSE.CuSparseMatrixCSR(dense64)

sparse32csc * dense32 # ERROR
dense32 * sparse32csc # NO ERROR
(sparse32csc' * dense32')' # ERROR

sparse64csc * dense64 # ERROR
dense64 * sparse64csc # NO ERROR
(dense64' * sparse64csc')' # NO ERROR

sparse64csr * dense64 # NO ERROR
dense64 * sparse64csr # ERROR
(sparse64csr' * dense64')' # NO ERROR