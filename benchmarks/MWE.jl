using CUDA
using SparseArrays

a = CUDA.rand(1, 1)
b = cu(sprand(Float64, 1, 1, 1.))

# works
b * a
a * b

# explodes for CUDA driver v12 but works for CUDA driver v11.4, tested on julia 1.9 and 1.10
c = CUSPARSE.CuSparseMatrixCSC{Float64, Int32}(b.colPtr, b.rowVal, b.nzVal, (1, 1))
d = CUDA.rand(Float64, 1, 1)

d * c
c * d

e = CUSPARSE.CuSparseMatrixCSR(c)
e * d

# also explodes for CUDA driver v11.4, on both versions of julia
d * e