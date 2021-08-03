@testset "PepsTensor correctly builds PEPS network" begin

m = 3
n = 4
t = 3

β = 1.0

L = m * n * t
T = Float64

instance = "$(@__DIR__)/instances/pathological/chim_$(m)_$(n)_$(t).txt"

ig = ising_graph(instance)


fg = factor_graph(
    ig,
    spectrum=full_spectrum,
    cluster_assignment_rule=super_square_lattice((m, n, t))
)

x, y = m, n

for transform ∈ all_lattice_transformations
    peps = PEPSNetwork(x, y, fg, transform, β=β)

    ψ = IdentityMPS()

    for i ∈ peps.nrows:-1:1
        ψ = MPO(T, peps, i, :up) * MPO(T, peps, i) * ψ
        @test MPS(peps, i) ≈ ψ
    end
end

end
