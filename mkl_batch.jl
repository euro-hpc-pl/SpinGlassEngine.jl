using MKL_jll, OpenBLAS_jll, Libdl, LinearAlgebra

const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const DGEMM_BATCH = Libdl.dlsym(libMKL, :dgemm_batch)

function dgemm_batch()
    group_count = 2
    group_size = fill(1, group_count)
    M, N, K = 32, 32, 32
    a_array = fill(rand(M, K), group_count)
    b_array = fill(rand(K, N), group_count)
    c_array = fill(Matrix{Float64}(undef, M, N), group_count)

    lda_array = stride.(a_array, Ref(2))
    ldb_array = stride.(b_array, Ref(2))
    ldc_array = stride.(c_array, Ref(2))

    m_array = fill(M, group_count)
    n_array = fill(N, group_count)
    k_array = fill(K, group_count)

    transa_array = fill('N', group_count)
    # transa_array = fill(UCint8(78), group_count)
    transb_array = fill('N', group_count)

    alpha_array = fill(one(Float64), group_count)
    beta_array = fill(zero(Float64), group_count)
    @time a_array .* b_array
    @time a_array .* b_array
    @show transa_array
    ccall(
        DGEMM_BATCH, Cvoid,
        (
            Ptr{Cchar}, Ptr{Cchar}, #transa_array, transb_array
            Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, #m_array, n_array, k_array
            Ptr{Float64}, #alpha_array
            Ptr{Float64}, Ptr{Cint}, #a_array, lda_array
            Ptr{Float64}, Ptr{Cint}, #b_array, ldb_array
            Ptr{Float64}, #beta_array
            Ptr{Float64}, Ptr{Cint}, #c_array, ldc_array
            Ref{Cint}, Ptr{Cint}, #group_count, group_size
        ),
        transa_array, transb_array,
        m_array, n_array, k_array,
        alpha_array,
        a_array, lda_array,
        b_array, ldb_array,
        beta_array,
        c_array, ldc_array,
        group_count, group_size
    )
end

dgemm_batch()
