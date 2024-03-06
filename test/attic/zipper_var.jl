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
    brute_force(ig, onGPU ? :GPU : :CPU, num_states=num_states)
end

m, n, t = 3, 3, 3

Dcut = 8
β = 0.5
tolV = 1E-16
tolS = 1E-16
max_sweeps = 1
indβ = 1
ITERS_SVD = 1
ITERS_VAR = 1
DTEMP_MULT = 2
MAX_STATES = 128
METHOD = :psvd_sparse #:psvd_sparse #:svd
DE = 16.0
δp = 1E-5*exp(-β * DE)

ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P4/CBFM-P/SpinGlass/001_sg.txt")

# cl_h = clustered_hamiltonian(
#     ig,
#     100,
#     spectrum=my_brute_force, #rm _gpu to use CPU
#     cluster_assignment_rule=pegasus_lattice((m, n, t))
# )

cl_h = clustered_hamiltonian(
    ig,
    spectrum=full_spectrum, #rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

cl_h = truncate_clustered_hamiltonian_1site_BP(cl_h, 1)

params = MpsParameters(Dcut, tolV, max_sweeps, tolS, ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
search_params = SearchParameters(MAX_STATES, δp)

Strategy = [Zipper, ]  # MPSAnnealing SVDTruncate
tran = all_lattice_transformations #LatticeTransformation((1, 2, 3, 4), true)
Layout = GaugesEnergy
Gauge = NoUpdate

indβ = 1

for s in Strategy
    for tran ∈ all_lattice_transformations #3,4,5,6,8
        net = PEPSNetwork{SquareCrossDoubleNode{Layout}, Sparse}(m, n, cl_h, tran)

        ctr = MpsContractor{s, Gauge}(net, [β/6, β/3, β/2, β], :graduate_truncate, params; onGPU=onGPU)
        sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
        println("Strategy ", s)
        println("Transform ", tran)
        println("sol ", sol)
        println("Schmidts ", schmidts)
    end
end
