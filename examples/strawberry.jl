using SpinGlassTensors
using SpinGlassNetworks
using SpinGlassEngine
using LinearAlgebra
using MKL
using JLD2

instance = "$(@__DIR__)/instances/strawberry-glass-2-small.h5"
output = "$(@__DIR__)/bench_results/rmf"

if !isdir(output)
	mkpath(output)
end

β = 0.5
LAYOUT = GaugesEnergy

GAUGE =  NoUpdate
STRATEGY = SVDTruncate
SPARSITY = Dense
graduate_truncation = :graduate_truncate

INDβ = [3,]
MAX_STATES = 128  # [64, 256, 1024]
BOND_DIM = 8  # [4, 8, ]

hamming_dist = 640
eng = 500

MAX_SWEEPS = 1
VAR_TOL = 1E-16
TOL_SVD = 1E-16
ITERS_SVD = 1
ITERS_VAR = 1
DTEMP_MULT = 2
METHOD = :svd
trans = rotation(0)
δp = 0.0

cl_h = potts_hamiltonian(instance, 320, 240)

params = MpsParameters(BOND_DIM, VAR_TOL, MAX_SWEEPS, TOL_SVD, ITERS_SVD, ITERS_VAR, DTEMP_MULT, METHOD)
search_params = SearchParameters(MAX_STATES, δp)
single = SingleLayerDroplets(eng, hamming_dist, :hamming, :RMF)

net = PEPSNetwork{SquareCrossSingleNode{LAYOUT}, SPARSITY}(320, 240, cl_h, trans)
ctr = MpsContractor{STRATEGY, GAUGE}(net, [β/6, β/3, β/2, β], graduate_truncation, params, onGPU=true)
merge_strategy = merge_branches(ctr, :nofit, single)
sol, schmidts = low_energy_spectrum(ctr, search_params, merge_strategy)

save_object(joinpath(output, "strawbery_sol_5.jld2"), sol)
save_object(joinpath(output, "strawbery_schmidts_5.jld2"), schmidts)

