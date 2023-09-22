using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using LinearAlgebra
using Logging

disable_logging(LogLevel(1))

m = 3
n = 4
t = 3

L = n * m * t
max_cl_states = 2^(t-0)

β = 0.5
bond_dim = 16
δp = 1E-3
num_states = 1000

instance = "$(@__DIR__)/../test/instances/pathological/chim_$(m)_$(n)_$(t).txt"

cl_h = clustered_hamiltonian(
    ising_graph(instance),
    max_cl_states,
    spectrum=brute_force,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 10)
search_params = SearchParameters(num_states, δp)

Strategy = SVDTruncate

for Layout ∈ (GaugesEnergy, ), transform ∈ rotation.([0])
    net = PEPSNetwork{SquareSingleNode{Layout}, Dense}(m, n, cl_h, transform, :rand)
    ctr = MpsContractor{Strategy}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)

    indβ = 3
    for i ∈ 1:m-1
        ψ_top = mps_top(ctr, i, indβ)
        ψ_bot = mps(ctr, i+1, indβ)
        overlap = tr(overlap_density_matrix(ψ_top, ψ_bot, indβ))
        overlap2 = ψ_bot * ψ_top
    end
    #clear_memoize_cache()
end
