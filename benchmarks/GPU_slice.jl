using BenchmarkTools
using CUDA


STDOUT = Base.stdout

function slice_gpu(M::AbstractArray{Float64, 3}, proj::AbstractArray{Int64,1}, len::Int64)
    B = permutedims(M[:, proj[1 : len], :], (1, 3, 2))
    B = CUDA.CuArray(B)
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

    println(".................................")
    println("CUDA -> permute dims:")
    println(".................................")
    println("mean")
    show(STDOUT,MIME"text/plain"(),mean(t1))
    println("\n")
    println("median")
    show(STDOUT,MIME"text/plain"(),median(t1))
    println("\n")
    println("------------------")

    t2 = @benchmark slice_gpu($B, $r_proj, $len_proj)

    println(".................................")
    println("permute dims -> CUDA:")
    println(".................................")
    println("mean")
    show(STDOUT,MIME"text/plain"(),mean(t2))
    println("\n")
    println("median")
    show(STDOUT,MIME"text/plain"(),median(t2))
    println("\n")
    println("------------------")
    CUDA.reclaim()
end