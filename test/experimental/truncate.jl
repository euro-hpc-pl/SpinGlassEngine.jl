using SpinGlassExhaustive
using SpinGlassEngine
using SpinGlassNetworks
using SpinGlassTensors
using Logging
using LightGraphs
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
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m, n, t = 3, 3, 3

Dcut = 8
β = 0.5
tolV = 1E-16
tolS = 1E-16
max_sweeps = 0
indβ = 1
ITERS_SVD = 2
ITERS_VAR = 1
DTEMP_MULT = 2
MAX_STATES = 128
METHOD = :psvd_sparse #:psvd_sparse #:svd
DE = 16.0
δp = 1E-5*exp(-β * DE)
ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P4/CBFM-P/SpinGlass/001_sg.txt")
results_folder = "$(@__DIR__)/../instances/pegasus_random/P4/CBFM-P/SpinGlass/BP"
inst = "001"
params = MpsParameters(Dcut, tolV, max_sweeps, tolS, ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
search_params = SearchParameters(MAX_STATES, δp)

Strategy = Zipper  # MPSAnnealing SVDTruncate
Layout = GaugesEnergy
Gauge = NoUpdate
cl_states = [2^10,]
iter = 2

for cs ∈ cl_states
    println("===================================")
    println("Cluster states ", cs)
    println("===================================")

    for tran ∈ [LatticeTransformation((1, 2, 3, 4), false),]#all_lattice_transformations
        println("===============")
        println("Transform ", tran)

        cl_h = clustered_hamiltonian(
            ig,
            spectrum= full_spectrum, #rm _gpu to use CPU
            cluster_assignment_rule=pegasus_lattice((m, n, t))
        )

        println("Truncate iter ", iter)
        #@time cl_h = truncate_clustered_hamiltonian_2site_energy(cl_h, cs)
        @time cl_h = truncate_clustered_hamiltonian(cl_h, β, cs, results_folder, inst; tol=1e-6, iter=iter)
        for v ∈ vertices(cl_h)
            println(v, " -> ", length(get_prop(cl_h, v, :spectrum).energies))
        end

        net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparse}(m, n, cl_h, tran)
        ctr = MpsContractor{Strategy, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)
        sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
        println("sol ", sol)
        # println("Schmidts ", schmidts)
    end
end
