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
β = 1.
tolV = 1E-16
tolS = 1E-16
max_sweeps = 4
indβ = 1
ITERS_SVD = 4
ITERS_VAR = 4
DTEMP_MULT = 2
MAX_STATES = 128
DE = 16.0
δp = 1E-5*exp(-β * DE)

ig = ising_graph("$(@__DIR__)/../instances/pegasus_random/P4/CBFM-P/SpinGlass/001_sg.txt")

fg = factor_graph(
    ig,
    100,
    spectrum=full_spectrum, #rm _gpu to use CPU
    cluster_assignment_rule=pegasus_lattice((m, n, t))
)

params = MpsParameters(Dcut, tolV, max_sweeps, tolS, ITERS_SVD, ITERS_VAR, DTEMP_MULT)
search_params = SearchParameters(MAX_STATES, δp)

Strategy = [MPSAnnealing, Zipper, ] #SVDTruncate
tran = LatticeTransformation((1, 2, 3, 4), true)
Layout = EnergyGauges
Gauge = NoUpdate

indβ = 1

net = PEPSNetwork{SquareStar2{Layout}, Sparse}(m, n, fg, tran)
for s in Strategy
    ctr = MpsContractor{s, Gauge}(net, [β], :graduate_truncate, params; onGPU=onGPU)
    sol, schmidts = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    println("Strategy ", s)
    println("sol ", sol)
    println("Schmidts ", schmidts)
end
