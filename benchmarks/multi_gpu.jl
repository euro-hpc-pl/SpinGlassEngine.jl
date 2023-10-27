# spawn one worker per device
using Distributed, CUDA
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassExhaustive


# This code generates WARNING: using LinearAlgebra.I in module Main conflicts with an existing identifier. I dont know why.


INSTANCE_DIR = "$(@__DIR__)/../test/instances/pegasus_random/P4/CBFM-P/SpinGlass" # remember to change it also in @everywhere
BETAS = [0.5,] #collect(0.2:0.1:1.0)
LAYOUT = (GaugesEnergy,)
TRANSFORM = all_lattice_transformations
BOND_DIM = [8, ]
MAX_SWEEPS = [0,]
I = [1,]

n_gpus = length(devices())
addprocs(n_gpus)

@everywhere begin
    using CUDA, LinearAlgebra
    using SpinGlassEngine
    using SpinGlassNetworks
    using SpinGlassTensors
    using SpinGlassExhaustive
    using Logging
    using CSV, DataFrames
    using Memoization
    using MKL

    disable_logging(LogLevel(1))

    function assign_resources(device_id, n_cores)
        CUDA.device!(device_id)
        BLAS.set_num_threads(n_cores)
    end

    function brute_force_gpu(ig::IsingGraph; num_states::Int)
        brute_force(ig, :GPU, num_states=num_states)
    end

    INSTANCE_DIR = "$(@__DIR__)/../test/instances/pegasus_random/P4/CBFM-P/SpinGlass"
    OUTPUT_DIR = "$(@__DIR__)/results/pegasus_random/P4/CBFM-P/new_zipper/truncate_2_8"

    if !Base.Filesystem.isdir(OUTPUT_DIR)
        Base.Filesystem.mkpath(OUTPUT_DIR)
    end

    M, N, T = 3, 3, 3
    GAUGE =  NoUpdate
    STRATEGY = Zipper #MPSAnnealing #SVDTruncate
    SPARSITY = Sparse
    graduate_truncation = :graduate_truncate
    MAX_STATES = 128
    DE = 16.0
    cs = 2^8

    VAR_TOL = 1E-16
    TOL_SVD = 1E-16
    ITERS_SVD = 2
    ITERS_VAR = 1
    DTEMP_MULT = 2
    METHOD = :psvd_sparse

    function pegasus_sim(inst, trans, β, Layout, bd, ms)
        δp = 1E-5*exp(-β * DE)
    
        cl_h = clustered_hamiltonian(
            ising_graph(INSTANCE_DIR * "/" * inst),
            spectrum=full_spectrum,
            cluster_assignment_rule=pegasus_lattice((M, N, T))
            )
        
        cl_h = truncate_clustered_hamiltonian_2site_energy(cl_h, cs)
    
        params = MpsParameters(bd, VAR_TOL, ms, TOL_SVD, ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
        search_params = SearchParameters(MAX_STATES, δp)
    
        net = PEPSNetwork{SquareCrossDoubleNode{Layout}, SPARSITY}(M, N, cl_h, trans)
        ctr = MpsContractor{STRATEGY, GAUGE}(net, [β/6, β/3, β/2, β], graduate_truncation, params)
        sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    
        cRAM = round(Base.summarysize(Memoization.caches) * 1E-9; sigdigits=2)
        clear_memoize_cache()
        sol, ctr, cRAM, schmidts
    end
    
    function run_bench(inst::String, β::Real, t, l, bd, ms, i)
        hash_name = hash(string(inst, β, t, l, bd, ms, i))
        out_path = string(OUTPUT_DIR, "/", hash_name, ".csv")
    
        if isfile(out_path)
            println("Skipping for $β, $t, $l, $bd, $ms.")
        else
            data = try
                tic_toc = @elapsed sol, ctr, cRAM, schmidts = pegasus_sim(inst, t, β, l, bd, ms)
            
                data = DataFrame(
                    :instance => inst,
                    :β => β,
                    :Layout => l,
                    :transform => t,
                    :energy => sol.energies[begin],
                    :probabilities => sol.probabilities,
                    :discarded_probability => sol.largest_discarded_probability,
                    :statistic => minimum(values(ctr.statistics)),
                    :max_states => MAX_STATES,
                    :bond_dim => bd,
                    :de => DE,
                    :max_sweeps => ms,
                    :cl_states => cs,
                    :iters_svd => ITERS_SVD,
                    :iters_var => ITERS_VAR,
                    :dtemp_mult => DTEMP_MULT,
                    :var_tol => VAR_TOL,
                    :time => tic_toc,
                    :cRAM => cRAM,
                    :schmidts => schmidts
                )
            catch err
                data = DataFrame(
                    :instance => inst,
                    :β => β,
                    :Layout => l,
                    :transform => t,
                    :max_states => MAX_STATES,
                    :cl_states => cs,
                    :iters_svd => ITERS_SVD,
                    :iters_var => ITERS_VAR,
                    :dtemp_mult => DTEMP_MULT,
                    :bond_dim => bd,
                    :de => DE,
                    :max_sweeps => ms,
                    :var_tol => VAR_TOL,
                    :error => err
                )
            end
            println(data)
            CSV.write(out_path, data, delim = ';', append = false)
        end #if
    end


end


asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        assign_resources(d, 4)
        c = BLAS.get_num_threads()
        println("Worker $p uses device $d and $c cores")
    end
end

all_params = collect(
    Iterators.product(
        readdir(INSTANCE_DIR, join=false), BETAS, TRANSFORM, LAYOUT, BOND_DIM, MAX_SWEEPS, I)
)

@distributed for parameters in all_params
    run_bench(parameters...)
    GC.gc()
end

rmprocs()
# After every use close julia, workers are left even so they shoud be deleted