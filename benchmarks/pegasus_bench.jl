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

M, N, T = 3, 3, 3
INSTANCE_DIR = "$(@__DIR__)/../test/instances/pegasus_random/P4/RAU/SpinGlass"
OUTPUT_DIR = "$(@__DIR__)/results/pegasus_random/P4/RAU"

BETAS =  [2,] #collect(0.5:0.5:3)
LAYOUT = (GaugesEnergy,)
TRANSFORM = all_lattice_transformations

GAUGE =  NoUpdate
STRATEGY = MPSAnnealing #SVDTruncate
SPARSITY = Sparse
graduate_truncation = :graduate_truncate

INDβ = [3,] #[1, 2, 3]
MAX_STATES = 128
BOND_DIM = 2
DE = 16.0

MAX_SWEEPS = 10
VAR_TOL = 1E-8
TOL_SVD = 1E-16
disable_logging(LogLevel(1))
BLAS.set_num_threads(1)

function pegasus_sim(inst, trans, β, Layout)
    max_cl_states = 2 ^ T
    δp = 1E-5*exp(-β * DE)

    cl_h = clustered_hamiltonian(
        ising_graph(INSTANCE_DIR * "/" * inst),
        spectrum=brute_force_gpu,
        cluster_assignment_rule=pegasus_lattice((M, N, T))
    )
    params = MpsParameters(BOND_DIM, VAR_TOL, MAX_SWEEPS, TOL_SVD)
    search_params = SearchParameters(MAX_STATES, δp)
  
    net = PEPSNetwork{SquareCrossDoubleNode{Layout}, SPARSITY}(M, N, cl_h, trans)
    ctr = MpsContractor{STRATEGY, GAUGE}(net, [β/6, β/3, β/2, β], graduate_truncation, params)
    sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

    cRAM = round(Base.summarysize(Memoization.caches) * 1E-9; sigdigits=2)
    clear_memoize_cache()
    sol, ctr, cRAM
end

function run_bench(inst::String, β::Real, t, l)
    hash_name = hash(string(inst, β, t, l))
    out_path = string(OUTPUT_DIR, "/", hash_name, ".csv")

    if isfile(out_path)
        println("Skipping for $β, $t, $l.")
    else
        data = try
            tic_toc = @elapsed sol, ctr, cRAM = pegasus_sim(inst, t, β, l)

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
                :bond_dim => BOND_DIM,
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
                :bond_dim => BOND_DIM,
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
        readdir(INSTANCE_DIR, join=false), BETAS, TRANSFORM, LAYOUT)
)

# for i ∈ (1+rank):size:length(all_params)
#     run_bench(all_params[i]...)
#     GC.gc()
# end
run_bench(all_params[1]...)