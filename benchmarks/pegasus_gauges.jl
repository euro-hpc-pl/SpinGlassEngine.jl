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
using JSON3

function brute_force_gpu(ig::IsingGraph; num_states::Int)
    brute_force(ig, :GPU, num_states=num_states)
end

MPI.Init()
size = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)

M, N, T = 3, 3, 3
INSTANCE_DIR = "$(@__DIR__)/../test/instances/pegasus_random/P4/RCO/SpinGlass/single"
OUTPUT_DIR = "$(@__DIR__)/results/pegasus_random/P4/RCO/gauges/final_bench_float64_bd8_betas_tr2^20_withgauges_betahalf"
if !Base.Filesystem.isdir(OUTPUT_DIR)
    Base.Filesystem.mkpath(OUTPUT_DIR)
end

BETAS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] #collect(0.5:0.5:3)
LAYOUT = (GaugesEnergy,)
TRANSFORM = all_lattice_transformations
TT = Float64
GAUGE =  GaugeStrategy
STRATEGY = Zipper
SPARSITY = Sparse
graduate_truncation = true

MAX_STATES = 1024
BOND_DIM = 8
DE = 16.0
cs=2^20
iter = 2

RESULTS_FOLDER = "$(@__DIR__)/../test/instances/pegasus_random/P4/RCO/BP"
if !Base.Filesystem.isdir(RESULTS_FOLDER)
    Base.Filesystem.mkpath(RESULTS_FOLDER)
end
MAX_SWEEPS = 0
VAR_TOL = 1E-16
TOL_SVD = 1E-16
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
METHOD = :psvd_sparse
disable_logging(LogLevel(1))
BLAS.set_num_threads(1)

function safe_filename(x)
    x_str = string(x)
    x_str = replace(x_str, r"[^\w\d]+" => "_")
    return x_str
end

function get_overlap_log_path(inst, β, t)
    short_inst = splitext(basename(inst))[1]
    trans_str = safe_filename(t)
    beta_str = replace(string(β), "." => "_")
    filename = "$(short_inst)_beta$(beta_str)_$(trans_str)_overlaps.csv"
    dir_path = joinpath(OUTPUT_DIR, "overlaps")
    mkpath(dir_path)
    return joinpath(dir_path, filename)
end

function pegasus_sim(inst, trans, β, Layout)
    δp = 0.0
    log_file = get_overlap_log_path(inst, β, trans)
    potts_h = potts_hamiltonian(
        ising_graph(INSTANCE_DIR * "/" * inst),
        spectrum=full_spectrum,
        cluster_assignment_rule=pegasus_lattice((M, N, T))
    )

    potts_h = truncate_potts_hamiltonian(potts_h, 0.5, cs, RESULTS_FOLDER, inst; tol=1e-6, iter=iter)

    params = MpsParameters{TT}(; bond_dim=BOND_DIM, var_tol=TT(VAR_TOL), num_sweeps=MAX_SWEEPS)

    search_params = SearchParameters(; max_states=MAX_STATES, cutoff_prob=δp)
  
    net = PEPSNetwork{SquareCrossDoubleNode{Layout}, SPARSITY, TT}(M, N, potts_h, trans)
    ctr = MpsContractor{STRATEGY, GAUGE, TT}(net, params; onGPU = true, beta = TT(β), graduate_truncation = graduate_truncation)
    update_gauges!(ctr, M, Val(:up), log_file)
    sol, schmidts = low_energy_spectrum(
        ctr,
        search_params,
        merge_branches(
            ctr;
            merge_prob = :none,
        ); 
    )
    clear_memoize_cache()
    sol, ctr, schmidts
end


function run_bench(inst::String, β::Real, t, l)
    hash_name = hash(string(inst, β, t, l))
    out_path = string(OUTPUT_DIR, "/", hash_name, ".csv")

    if isfile(out_path)
        println("Skipping for $β, $t, $l.")
    else
        data = try
            tic_toc = @elapsed sol, ctr, s = pegasus_sim(inst, t, β, l)

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
                :time => tic_toc,
                :schmidts => s
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