using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Memoize
using NLsolve

m = 4
n = 4
t = 8

L = n * m * t
max_cl_states = 2^(t-0)

β = 0.5
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

Strategy = SVDTruncate
@testset "Updating gauges works correctly." begin
    for Layout ∈ (GaugesEnergy, ), transform ∈ rotation.([0])

        net = PEPSNetwork{Square{Layout}, Dense}(m, n, fg, transform, :id)
        ctr = MpsContractor{Strategy}(net, [β/8, β/4, β/2, β], params)

        @testset "Overlaps calculated differently are the same." begin
            indβ = 3
            for i ∈ 1:m-1
                ψ_top = mps_top(ctr, i, indβ)
                ψ_bot = mps(ctr, i+1, indβ)
                overlap = tr(overlap_density_matrix(ψ_top, ψ_bot, indβ))
                @test overlap ≈ ψ_bot * ψ_top
            end
        end

        @testset "Numerical gauge optimization." begin
            b, t  = rand(m), rand(m)
            b ./= sum(b)
            t ./= sum(t)
            x0 = rand(length(b))

            function fun!(F, x)
                y = 1.0 ./ x
                z = (t .* x) .* dot(b, y) .- (b .* y) .* dot(b, x)
                for (i, e) ∈ enumerate(z) F[i] = e end
            end
            r = nlsolve(fun!, x0)
            @test converged(r)

        end

        clear_memoize_cache()
    end
end
