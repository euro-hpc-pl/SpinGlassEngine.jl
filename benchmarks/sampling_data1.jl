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

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

MPI.Init()
size = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

M, N, T = 4, 4, 4
BETAS = collect(0.6:0.1:2.0)
i = BETAS[1]
INSTANCE_DIR = "$(@__DIR__)/../test/instances/square_gauss/S8"
OUTPUT_DIR = "$(@__DIR__)/results/square_gauss/S8/beta$(i)"

LAYOUT = (GaugesEnergy,)
TRANSFORM = [rotation(0),] #all_lattice_transformations

GAUGE =  NoUpdate
STRATEGY = SVDTruncate
SPARSITY = Sparse
graduate_truncation = :graduate_truncate

INDβ = [3,] #[1, 2, 3]
MAX_STATES = 10000
BOND_DIM = [16,]
DE = 16.0

MAX_SWEEPS = 50
VAR_TOL = 1E-16
TOL_SVD = 1E-16
disable_logging(LogLevel(1))
BLAS.set_num_threads(1)

onGPU = false

function sampling_sim(inst, trans, β, Layout, bd)
    δp = 1E-5*exp(-β * DE)

    fg = factor_graph(
        ising_graph(INSTANCE_DIR * "/" * inst),
        #2^max_cl_states,
        spectrum=my_brute_force,
        cluster_assignment_rule=periodic_lattice((M, N, T))
        )
    params = MpsParameters(bd, VAR_TOL, MAX_SWEEPS, TOL_SVD)
    search_params = SearchParameters(MAX_STATES, δp)
  
    net = PEPSNetwork{Square{Layout}, SPARSITY}(M, N, fg, trans)
    ctr = MpsContractor{STRATEGY, GAUGE}(net, [β/6, β/3, β/2, β], graduate_truncation, params; onGPU=onGPU)
    sol = gibbs_sampling(ctr, search_params, merge_branches(ctr))
    st = decode_samples(ctr, sol)

    cRAM = round(Base.summarysize(Memoization.caches) * 1E-9; sigdigits=2)
    clear_memoize_cache()
    sol, st, ctr, cRAM
end

function run_bench(inst::String, β::Real, t, l, bd)
    hash_name = hash(string(inst, β, t, l, bd))
    out_path = string(OUTPUT_DIR, "/", hash_name, ".csv")

    if isfile(out_path)
        println("Skipping for $β, $t, $l, $bd.")
    else
        data = try
            tic_toc = @elapsed sol, st, ctr, cRAM = sampling_sim(inst, t, β, l, bd)
            data = DataFrame(
                :instance => inst,
                :β => β,
                :Layout => l,
                :transform => t,
                :energy => sol.energies,
                :sol_states => sol.states,
                :states => st,
                :probabilities => sol.probabilities,
                :discarded_probability => sol.largest_discarded_probability,
                :statistic => minimum(values(ctr.statistics)),
                :max_states => MAX_STATES,
                :bond_dim => bd,
                :de => DE,
                :max_sweeps => MAX_SWEEPS,
                :var_tol => VAR_TOL,
                :time => tic_toc,
                :cRAM => cRAM
            )
        catch err
            data = DataFrame(
                :instance => inst,
                :β => β,
                :Layout => l,
                :transform => t,
                :max_states => MAX_STATES,
                :bond_dim => bd,
                :de => DE,
                :max_sweeps => MAX_SWEEPS,
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
        readdir(INSTANCE_DIR, join=false), BETAS, TRANSFORM, LAYOUT, BOND_DIM)
)

for i ∈ (1+rank):size:length(all_params)
    run_bench(all_params[i]...)
    GC.gc()
end
#run_bench(all_params[1]...)