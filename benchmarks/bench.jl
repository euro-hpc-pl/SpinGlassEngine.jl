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
INSTANCE_DIR = "$(@__DIR__)/../test/instances/pegasus_random/P4/RCO/SpinGlass"
OUTPUT_DIR = "$(@__DIR__)/results/pegasus_random/P4/RCO/final_bench_float32_betas_tr2^20"
if !Base.Filesystem.isdir(OUTPUT_DIR)
    Base.Filesystem.mkpath(OUTPUT_DIR)
end
BETAS =  [0.25,0.5, 1.0, 2.0, 5.0, 10.0] #collect(0.5:0.5:3)
LAYOUT = (GaugesEnergy,)
TRANSFORM = [all_lattice_transformations[1],]
TT = Float32
GAUGE =  NoUpdate
STRATEGY = Zipper #SVDTruncate
SPARSITY = Sparse
graduate_truncation = :graduate

MAX_STATES = 1024
BOND_DIM = 8
DE = 16.0
cs=2^20
iter = 2

RESULTS_FOLDER = "$(@__DIR__)/../test/instances/pegasus_random/P4/RCO/BP"
inst="001"
MAX_SWEEPS = 0
VAR_TOL = 1E-16
TOL_SVD = 1E-16
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
METHOD = :psvd_sparse
disable_logging(LogLevel(1))
BLAS.set_num_threads(1)

function pegasus_sim(inst, trans, β, Layout)
    δp = 0.0

    potts_h = potts_hamiltonian(
        ising_graph(INSTANCE_DIR * "/" * inst),
        spectrum=full_spectrum,
        cluster_assignment_rule=pegasus_lattice((M, N, T))
    )
    potts_h = truncate_potts_hamiltonian(potts_h, β, cs, RESULTS_FOLDER, inst; tol=1e-6, iter=iter)

    # params = MpsParameters{TT}(;bond_dim=BOND_DIM, var_tol=TT(VAR_TOL), num_sweeps=MAX_SWEEPS, tol_SVD=TT(TOL_SVD), iters_SVD=ITERS_SVD, iters_var=ITERS_VAR, Dtemp_multiplier=DTEMP_MULT, method=METHOD)
    params = MpsParameters{TT}(; bond_dim=BOND_DIM, var_tol=TT(VAR_TOL), num_sweeps=MAX_SWEEPS)

    search_params = SearchParameters(; max_states=MAX_STATES, cut_off_prob=δp)
  
    net = PEPSNetwork{SquareCrossDoubleNode{Layout}, SPARSITY, TT}(M, N, potts_h, trans)
    ctr = MpsContractor{STRATEGY, GAUGE, TT}(net, params; onGPU = true, beta = TT(β), graduate_truncation = graduate_truncation)
    sol, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr; merge_type=:nofit))

    # cRAM = round(Base.summarysize(Memoization.caches) * 1E-9; sigdigits=2)
    clear_memoize_cache()
    sol, ctr
end

function run_bench(inst::String, β::Real, t, l)
    hash_name = hash(string(inst, β, t, l))
    out_path = string(OUTPUT_DIR, "/", hash_name, ".csv")

    if isfile(out_path)
        println("Skipping for $β, $t, $l.")
    else
        data = try
            tic_toc = @elapsed sol, ctr = pegasus_sim(inst, t, β, l)

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
                :cs => cs,
                :de => DE,
                :max_sweeps => MAX_SWEEPS,
                :var_tol => VAR_TOL,
                :time => tic_toc
                # :cRAM => cRAM
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

for i ∈ (1+rank):size:length(all_params)
    run_bench(all_params[i]...)
    GC.gc()
end
# run_bench(all_params[1]...)