using MPI
using LinearAlgebra
using MKL
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassExhaustive
using Logging
using CSV
using DataFrames
using Memoization

function brute_force_gpu(ig::IsingGraph; num_states::Int)
    brute_force(ig, :GPU, num_states=num_states)
end

MPI.Init()
size = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

M, N, T = 7, 7, 3
INSTANCE_DIR = "$(@__DIR__)/../test/instances/pegasus_random/P8/CBFM-P/SpinGlass/single"
OUTPUT_DIR = "$(@__DIR__)/results/pegasus_random/P8/CBFM-P/new_zipper/GE_truncate_SVDTruncate"

if !Base.Filesystem.isdir(OUTPUT_DIR)
    Base.Filesystem.mkpath(OUTPUT_DIR)
end

BETAS = [0.6,] #collect(0.2:0.1:1.0)
LAYOUT = (GaugesEnergy,) #GaugesEnergy - stabilne, EnergyGauges - niestabilne
TRANSFORM = all_lattice_transformations 

GAUGE =  NoUpdate
STRATEGY = [SVDTruncate, ] #MPSAnnealing #SVDTruncate
SPARSITY = Sparse
graduate_truncation = :graduate_truncate

INDβ = [3,] #[1, 2, 3]
MAX_STATES = 128
BOND_DIM = [8,]
DE = 16.0
#MAX_CL = [2,4,6,8,10,12]

MAX_SWEEPS = [1,]
VAR_TOL = 1E-16
TOL_SVD = 1E-16
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
METHOD = [:psvd_sparse, :svd,] #:psvd_sparse #:svd

I = [1,]
disable_logging(LogLevel(1))
BLAS.set_num_threads(1)
CL_STATES = [2^8,]


function pegasus_sim(inst, trans, β, Layout, bd, ms, st, met, cs)
    δp = 1E-5*exp(-β * DE)

    cl_h = clustered_hamiltonian(
        ising_graph(INSTANCE_DIR * "/" * inst),
        spectrum=full_spectrum,
        cluster_assignment_rule=pegasus_lattice((M, N, T))
        )

    cl_h = truncate_clustered_hamiltonian_2site_energy(cl_h, cs)

    params = MpsParameters(bd, VAR_TOL, ms, TOL_SVD, ITERS_SVD, ITERS_VAR, DTEMP_MULT, met)
    search_params = SearchParameters(MAX_STATES, δp)

    net = PEPSNetwork{SquareCrossDoubleNode{Layout}, SPARSITY}(M, N, cl_h, trans)
    ctr = MpsContractor{st, GAUGE}(net, [β/6, β/3, β/2, β], graduate_truncation, params)
    sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

    cRAM = round(Base.summarysize(Memoization.caches) * 1E-9; sigdigits=2)
    clear_memoize_cache()
    sol, ctr, cRAM, schmidts
end

function run_bench(inst::String, β::Real, t, l, bd, ms, st, met, cs, i)
    hash_name = hash(string(inst, β, t, l, bd, ms, st, met, cs, i))
    out_path = string(OUTPUT_DIR, "/", hash_name, ".csv")

    if isfile(out_path)
        println("Skipping for $β, $t, $l, $bd, $ms, $st, $met, $cs.")
    else
        data = try
            tic_toc = @elapsed sol, ctr, cRAM, schmidts = pegasus_sim(inst, t, β, l, bd, ms, st, met, cs)

            data = DataFrame(
                :instance => inst,
                :β => β,
                :Layout => l,
                :Strategy => st,
                :transform => t,
                :method => met,
                :cluster_states => cs,
                :energy => sol.energies[begin],
                :probabilities => sol.probabilities,
                :discarded_probability => sol.largest_discarded_probability,
                :statistic => minimum(values(ctr.statistics)),
                :max_states => MAX_STATES,
                :bond_dim => bd,
                :de => DE,
                :max_sweeps => ms,
                :iters_svd => ITERS_SVD,
                :iters_var => ITERS_VAR,
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
                :Strategy => st,
                :transform => t,
                :method => met,
                :cluster_states => cs,
                :max_states => MAX_STATES,
                :iters_svd => ITERS_SVD,
                :iters_var => ITERS_VAR,
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

all_params = collect(
    Iterators.product(
        readdir(INSTANCE_DIR, join=false), BETAS, TRANSFORM, LAYOUT, BOND_DIM, MAX_SWEEPS, STRATEGY, METHOD, CL_STATES, I)
)

for i ∈ (1+rank):size:length(all_params)
    run_bench(all_params[i]...)
    GC.gc()
end
#run_bench(all_params[1]...)