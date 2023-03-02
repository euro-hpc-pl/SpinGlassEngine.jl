using LinearAlgebra
using BenchmarkTools
using TensorCast
import SpinGlassTensors: format_bytes

# random 
a = @allocated X = rand(Float64, (1000, 1000, 1000))
println("base: ", format_bytes(a))

# @view
a = @allocated X[1:100, :, 1:500]
b = @allocated @view X[1:100, :, 1:500]
println("without @view: ", format_bytes(a))
println("with @view: ", format_bytes(b))
# view doesn't copy array, but using just [] does

# permutedims
a = @allocated X
b = @allocated permutedims(X, (2, 1, 3))
println("just matrix: ", format_bytes(a))
println("permutedims: ", format_bytes(b))
# permutedims copies array, permutedims! by definition copies array

a = @allocated @cast X[x, (y, z)] := X[x, y, z] 
b = @allocated @cast X[x, y, z] := X[x, (y, z)] y in 1:1000
c = @allocated @cast Y[y, z, x] := X[x, y, z]
println("@cast conecting legs: ", format_bytes(a))
println("@cast dividing legs: ", format_bytes(b))
println("@cast with reshape: ", format_bytes(c))

#cast doesn't copy, even when reshaping