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

# cast
a = @allocated @cast X[x, (y, z)] := X[x, y, z] 
b = @allocated @cast X[x, y, z] := X[x, (y, z)] y in 1:1000
c = @allocated @cast Y[y, z, x] := X[x, y, z]
println("@cast conecting legs: ", format_bytes(a))
println("@cast dividing legs: ", format_bytes(b))
println("@cast with permutedims: ", format_bytes(c))

#cast doesn't copy, even when reshaping

#reshape
a = @allocated reshape(X, (:, 10, 1000, 1000))
b = @allocated @cast X[i, j, y, z] := X[(i, j), y, z] j in 1:10
println("reshape: ", format_bytes(a))
println("reshape by @cast: ", format_bytes(b))

#neither of them seems to be coping full array, but @cast is more afective
