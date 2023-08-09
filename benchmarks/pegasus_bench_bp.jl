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
OUTPUT_DIR = "$(@__DIR__)/results/pegasus_random/P8/CBFM-P/BP/P8_truncate2site_2_16_20_i1-10_iters012"

if !Base.Filesystem.isdir(OUTPUT_DIR)
    Base.Filesystem.mkpath(OUTPUT_DIR)
end

BETAS = [0.5,] #collect(0.2:0.1:1.0)
LAYOUT = (GaugesEnergy,)
TRANSFORM = all_lattice_transformations 

GAUGE =  NoUpdate
STRATEGY = Zipper #MPSAnnealing #SVDTruncate
SPARSITY = Sparse
graduate_truncation = :graduate_truncate

INDβ = [3,] #[1, 2, 3]
MAX_STATES = 128
BOND_DIM = [8, ]
DE = 16.0
CL_STATES = [2^16, 2^20]
ITER_BP = [0, 2]
MAX_SWEEPS = [0,]
VAR_TOL = 1E-16
TOL_SVD = 1E-16
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
METHOD = :psvd_sparse
I = [1,]
disable_logging(LogLevel(1))
BLAS.set_num_threads(1)

function create_fg(inst, β, ibp)
    tol = 1e-6
    fg = factor_graph(
        ising_graph(INSTANCE_DIR * "/" * inst),
        spectrum=full_spectrum,
        cluster_assignment_rule=pegasus_lattice((M, N, T))
        )
    new_fg = factor_graph_2site(fg, β)
    belief_propagation(new_fg, β; tol=tol, iter=ibp), fg
end

function pegasus_sim(inst, fg, beliefs, trans, β, Layout, bd, ms, cs, ibp)
    δp = 1E-5*exp(-β * DE)

    fg = truncate_factor_graph_2site_BP(fg, beliefs, cs; beta=β)

    params = MpsParameters(bd, VAR_TOL, ms, TOL_SVD, ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
    search_params = SearchParameters(MAX_STATES, δp)

    net = PEPSNetwork{SquareStar2{Layout}, SPARSITY}(M, N, fg, trans)
    ctr = MpsContractor{STRATEGY, GAUGE}(net, [β/6, β/3, β/2, β], graduate_truncation, params)
    sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr))

    cRAM = round(Base.summarysize(Memoization.caches) * 1E-9; sigdigits=2)
    clear_memoize_cache()
    sol, ctr, cRAM, schmidts
end

function run_bench(inst::String, β::Real, t, l, bd, ms, cs, ibp, i, fg, beliefs)
    hash_name = hash(string(inst, β, t, l, bd, ms, cs, ibp, i))
    out_path = string(OUTPUT_DIR, "/", hash_name, ".csv")

    if isfile(out_path)
        println("Skipping for $β, $t, $l, $bd, $ms, $cs, $ibp.")
    else
        data = try  
            tic_toc = @elapsed sol, ctr, cRAM, schmidts = pegasus_sim(inst, fg, beliefs, t, β, l, bd, ms, cs, ibp)
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
                :iter_bp => ibp,
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
                :iter_bp => ibp,
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

all_params = collect(
    Iterators.product(
        readdir(INSTANCE_DIR, join=false), BETAS, TRANSFORM, LAYOUT, BOND_DIM, MAX_SWEEPS, ITER_BP, I)
)

for ii ∈ (1+rank):size:length(all_params)
    inst, β, t, l, bd, ms, ibp, i = all_params[ii]
    beliefs, fg = create_fg(inst, β, ibp)
    for cs in CL_STATES
        run_bench(inst, β, t, l, bd, ms, cs, ibp, i, fg, beliefs)
        GC.gc()
    end
end
#run_bench(all_params[1]...)