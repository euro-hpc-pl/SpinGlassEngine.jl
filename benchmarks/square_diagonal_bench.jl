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

M, N, T = 50,50, 1
INSTANCE_DIR = "$(@__DIR__)/../test/instances/square/square_50x50/single"
OUTPUT_DIR = "$(@__DIR__)/results/square/50x50x1"
if !Base.Filesystem.isdir(OUTPUT_DIR)
    Base.Filesystem.mkpath(OUTPUT_DIR)
end

BETAS = 2.0
LAYOUT = (GaugesEnergy, )
TRANSFORM = all_lattice_transformations

GAUGE =  NoUpdate
STRATEGY = (Zipper,)
SPARSITY = (Sparse, )
graduate_truncation = :graduate_truncate

INDβ = [3,] #[1, 2, 3]
MAX_STATES = [256,]
BOND_DIM = [8,]
# cs = 2^12
# iter = 1

MAX_SWEEPS = [0,]
VAR_TOL = 1E-16
TOL_SVD = 1E-16
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
METHOD = :psvd_sparse
eng = 80
hamming_dist = 156 #25 #56 #100
I = [1,]

disable_logging(LogLevel(1))
BLAS.set_num_threads(1)

function pegasus_sim(inst, trans, β, Layout, st, sp, bd, ms, eng, hamming_dist, mstates)
    δp = 0.0
    
    cl_h = clustered_hamiltonian(
        ising_graph(INSTANCE_DIR * "/" * inst),
        spectrum = full_spectrum,
        cluster_assignment_rule=super_square_lattice((M, N, T))
    )

    params = MpsParameters(bd, VAR_TOL, ms, TOL_SVD, ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
    search_params = SearchParameters(mstates, δp)

    net = PEPSNetwork{SquareCrossSingleNode{Layout}, sp}(M, N, cl_h, trans)
    ctr = MpsContractor{st, GAUGE}(net, [β/6, β/3, β/2, β], graduate_truncation, params)
    clusters = split_into_clusters(ising_graph(INSTANCE_DIR * "/" * inst), super_square_lattice((M, N, T)))
    # sol1, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :fit, SingleLayerDroplets(eng, hamming_dist, :hamming)))
    sol1, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit, SingleLayerDroplets(eng, hamming_dist, :hamming)))

    sol2 = unpack_droplets(sol1, β)
    ig_states = decode_clustered_hamiltonian_state.(Ref(cl_h), sol2.states)

    ldrop = length(sol2.states)
    cRAM = round(Base.summarysize(Memoization.caches) * 1E-9; sigdigits=2)
    clear_memoize_cache()
    sol1, ctr, cRAM, schmidts, ldrop, sol2, ig_states, clusters
end

function run_bench(inst::String, β::Real, t, l, st, sp, bd, ms, eng, hamming_dist, mstates, i)
    hash_name = hash(string(inst, β, t, l, st, sp, bd, ms, mstates, i))
    out_path = string(OUTPUT_DIR, "/", hash_name, ".json")

    if isfile(out_path)
        println("Skipping for $β, $t, $l, $st, $sp, $bd, $ms, $mstates.")
    else
        data = try
            tic_toc = @elapsed sol, ctr, cRAM, schmidts, ldrop, droplets, ig_states, clusters = pegasus_sim(inst, t, β, l, st, sp, bd, ms, eng, hamming_dist, mstates)
            data = DataFrame(
                :instance => inst,
                :β => β,
                :Layout => l,
                :Strategy => st,
                :Sparsity => sp,
                :transform => t,
                :energy => sol.energies,
                :states => sol.states,
                :ig_states => [ig_states],
                :clusters => [clusters],
                :probabilities => sol.probabilities,
                :discarded_probability => sol.largest_discarded_probability,
                :drop_eng => [droplets.energies],
                :drop_states => [droplets.states],
                :drop_prob => [droplets.probabilities],
                :drop_degeneracy => [droplets.degeneracy],
                :drop_ldp => [droplets.largest_discarded_probability],
                :drop_number => ldrop,
                :statistic => minimum(values(ctr.statistics)),
                :max_states => mstates,
                :bond_dim => bd,
                :max_sweeps => ms,
                # :cl_states => cs,
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
                :Strategy => st,
                :Sparsity => sp,
                :transform => t,
                :max_states => mstates,
                # :cl_states => cs,
                :iters_svd => ITERS_SVD,
                :iters_var => ITERS_VAR,
                :dtemp_mult => DTEMP_MULT,
                :bond_dim => bd,
                :max_sweeps => ms,
                :var_tol => VAR_TOL,
                :error => err
            )
        end
        # println(data)
        # CSV.write(out_path, data, delim = ';', append = false)

        json_data = JSON3.write(data)
        # Write the JSON data to a file
        open(out_path, "w") do io
            print(io, json_data)
        end

    end #if
end

all_params = collect(
    Iterators.product(
        readdir(INSTANCE_DIR, join=false), BETAS, TRANSFORM, LAYOUT, STRATEGY, SPARSITY, BOND_DIM, MAX_SWEEPS, eng, hamming_dist, MAX_STATES, I)
)

for i ∈ (1+rank):size:length(all_params)
    run_bench(all_params[i]...)
    GC.gc()
end
#run_bench(all_params[1]...)