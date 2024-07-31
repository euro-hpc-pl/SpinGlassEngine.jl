using SpinGlassExhaustive
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using Graphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using Statistics
using LowRankApprox
using CUDA

disable_logging(LogLevel(1))
CUDA.allowscalar(false)

onGPU = true

function my_brute_force(ig::IsingGraph; num_states::Int)
    brute_force(ig, onGPU ? :GPU : :CPU, num_states = num_states)
end

m, n, t = 3, 3, 3

Dcut = 8
β = 0.5
tolV = 1E-16
tolS = 1E-16
max_sweeps = 0
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
MAX_STATES = 128
METHOD = :psvd_sparse #:psvd_sparse #:svd
DE = 16.0
δp = 1E-5 * exp(-β * DE)
ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P4/CBFM-P/SpinGlass/001_sg.txt")
results_folder = "$(@__DIR__)/../instances/pegasus_random/P4/CBFM-P/SpinGlass/BP"
inst = "001"
params =
    MpsParameters(Dcut, tolV, max_sweeps, tolS, ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
search_params = SearchParameters(MAX_STATES, δp)

Strategy = Zipper  # SVDTruncate
Layout = GaugesEnergy
Gauge = NoUpdate
cl_states = [2^10]
iter = 2

for cs ∈ cl_states
    println("===================================")
    println("Cluster states ", cs)
    println("===================================")

    for tran ∈ [LatticeTransformation((1, 2, 3, 4), false)]#all_lattice_transformations
        println("===============")
        println("Transform ", tran)

        potts_h = potts_hamiltonian(
            ig,
            spectrum = full_spectrum, #rm _gpu to use CPU
            cluster_assignment_rule = pegasus_lattice((m, n, t)),
        )

        println("Truncate iter ", iter)
        #@time potts_h = truncate_potts_hamiltonian_2site_energy(potts_h, cs)
        @time potts_h = truncate_potts_hamiltonian(
            potts_h,
            β,
            cs,
            results_folder,
            inst;
            tol = 1e-6,
            iter = iter,
        )
        for v ∈ vertices(potts_h)
            println(v, " -> ", length(get_prop(potts_h, v, :spectrum).energies))
        end

        net = PEPSNetwork{SquareCrossDoubleNode{Layout},Sparse}(m, n, potts_h, tran)
        ctr = MpsContractor{Strategy,Gauge}(
            net,
            params;
            onGPU = onGPU,
            beta = β,
            graduate_truncation = :graduate,
        )
        sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
        println("sol ", sol)
        # println("Schmidts ", schmidts)
    end
end
