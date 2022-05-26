using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Memoize

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t-0)

β = 1
bond_dim = 16
δp = 1E-3
num_states = 1000

instance = "$(@__DIR__)/instances/chimera_droplets/128power/001.txt"

fg = factor_graph(
    ising_graph(instance),
    max_cl_states,
    spectrum=brute_force,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

params = MpsParameters(bond_dim, 1E-8, 10)
search_params = SearchParameters(num_states, δp)
Gauge = NoUpdate

@testset "Updating gauges works correctly." begin
for Sparsity ∈ (Dense,)# Sparse) #MPSAnnealing
    # MPSAnnealing MethodError: no method matching mps_top(::MpsContractor{MPSAnnealing}, ::Int64, ::Int64)
    for Layout ∈ (EnergyGauges,)# GaugesEnergy, EngGaugesEng)
        for Lattice ∈ (Square, SquareStar), transform ∈ all_lattice_transformations[[1]]
            net = PEPSNetwork{Lattice{Layout}, Sparsity}(m, n, fg, transform, :id)
            ctr_svd = MpsContractor{SVDTruncate, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)
            ctr_anneal = MpsContractor{MPSAnnealing, Gauge}(net, [β/8, β/4, β/2, β], :graduate_truncate, params)

            @testset "Overlaps calculated differently are the same." begin
                indβ = 3
                for i ∈ 1:m-1
                    ψ_top = mps_top(ctr_svd, i, indβ)
                    ϕ_top = mps_top(ctr_anneal, i, indβ)
                    @test ψ_top * ψ_top ≈ 1
                    @test ϕ_top * ϕ_top ≈ 1
                    @test ψ_top * ϕ_top ≈ 1
                end
            end
            clear_memoize_cache()
        end
    end
end
end