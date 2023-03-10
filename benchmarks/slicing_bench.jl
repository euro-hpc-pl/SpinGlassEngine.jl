using BenchmarkTools, LinearAlgebra
import SpinGlassTensors: format_bytes
using CUDA

X = CUDA.rand(Float64, (1000, 1000, 1000))
#X1p = CuArray{Float64}(undef, (1000, 1000, 300))
X2p = CuArray{Float64}(undef, (1000, 1000, 300))
p = CuArray(rand(1:50, 1000))
vp = @view p[1:300]

#using normal slicing always creates a copy, even when using prealocated memory
a = @allocated X1p = X[:, :, vp]
println("allocation using normal slicing: ", format_bytes(a))

@time for kk = 1:20
    X1p = X[:, :, vp]
end

#using view doesn't create copy, and we must use copy! function to use preallocated memory
b = @allocated copy!(X2p, @view X[:, :, vp])
println("allocation using @view: ", format_bytes(b))

@time for kk = 1:20
    @views copy!(X2p, X[:, :, vp])
end

# #changes to original tensor don't translate into allocated
# println("original X: ",  X[1:2,1:2,1])
# X[1:2,1:2,1] = [100 100; 100 100]
# println("changed X: ", X[1:2,1:2,1])
# println("X preallocated: ", X2p[1:2,1:2,1])

# @profile bb(X2p, X, vp)
# pprof(flamegraph(); webhost = "localhost", webport = 57321)