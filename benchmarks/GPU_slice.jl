using BenchmarkTools
using CUDA


STDOUT = Base.stdout

function slice_gpu(M::AbstractArray{Float64, 3}, proj::AbstractArray{Int64,1}, len::Int64)
    B = CUDA.CuArray(M[:, proj[1 : len], :])
    B = permutedims(B, (1, 3, 2))
    B
end

for rg ∈ [5, 50, 500]
    len_proj = 2*rg
    println(len_proj)
    println("===================================")

    r_proj = sort(rand(1:rg, len_proj))
    A = rand(Float64, (len_proj, len_proj, len_proj))
    B = copy(A)

    t1 = @benchmark permutedims(CUDA.CuArray($A[:, $r_proj[1 : $len_proj], :]), (1, 3, 2))

    show(STDOUT,MIME"text/plain"(),median(t1))
    println("\n")
    println("------------------")

    t2 = @benchmark slice_gpu($B, $r_proj, $len_proj)

    show(STDOUT,MIME"text/plain"(),median(t2))
    println("\n")
    println("------------------")
    CUDA.reclaim()
end