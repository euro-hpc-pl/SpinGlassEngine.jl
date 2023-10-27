using LinearAlgebra
using MKL
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using CSV
using DataFrames
using Memoization
using Distributed

#M, N, T = 8, 8, 8
#INSTANCE_DIR = "$(@__DIR__)/instances/chimera_droplets/512power"
#OUTPUT_DIR = "$(@__DIR__)/results/512power/tmp"

M, N, T = 12, 12, 8
INSTANCE_DIR = "$(@__DIR__)/instances/chimera_droplets/1152power"
OUTPUT_DIR = "$(@__DIR__)/results/1152power/tmp"

BETAS = collect(2:2:14)
LAYOUT = (EnergyGauges, GaugesEnergy, EngGaugesEng)
TRANSFORM = all_lattice_transformations

GAUGE = NoUpdate
STRATEGY = SVDTruncate
SPARSITY = Dense

MAX_STATES = 1000
BOND_DIM = 32
DE = 1.0

MAX_SWEEPS = 10
VAR_TOL = 1E-8

disable_logging(LogLevel(1))
BLAS.set_num_threads(1)

function chimera_sim(inst, trans, β, Layout)
    max_cl_states = 2 ^ T
    δp = 1E-5 * exp(-β * DE)

    cl_h = clustered_hamiltonian(
        ising_graph(INSTANCE_DIR * "/" * inst),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((M, N, T))
    )
    params = MpsParameters(BOND_DIM, VAR_TOL, MAX_SWEEPS)
    search_params = SearchParameters(MAX_STATES, δp)

    net = PEPSNetwork{SquareSingleNode{Layout}, SPARSITY}(M, N, cl_h, trans)
    ctr = MpsContractor{STRATEGY, GAUGE}(net, [β/6, β/3, β/2, β], :graduate_truncate, params)
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
            tic_toc = @elapsed sol, ctr, cRAM = chimera_sim(inst, t, β, l)

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

println("Processors $(procs()) available.")

Distributed.pmap(
    p->run_bench(p...),
    Iterators.product(readdir(INSTANCE_DIR, join=false), BETAS, TRANSFORM, LAYOUT);
    distributed=true
)
