using TensorOperations
using SparseArrayKit
using CUDA

function randn_sparse(T::Type{<:Number}, sz::Dims, p = 0.5)
    a = SparseArray{T}(undef, sz)
    for I in keys(a)
        if rand() < p
            #a[I] = CUDA.randn(T)
            a[I] = randn(T)
        end
    end
    return a
end

T = Float64
A = randn_sparse(T, (3, 2, 3))
B = randn_sparse(T, (3, 3))

@tensor C[x, y, t] := B[z, t] * A[x, y, z]

nothing
