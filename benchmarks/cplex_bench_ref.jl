using LinearAlgebra
using CPLEX, JuMP
using SpinGlassNetworks, LightGraphs

qubo(h::Vector{T}, J::Matrix{T}) where T = 4 .* J .+ 2 .* Diagonal((h .- sum(J)))

instance = "$(@__DIR__)/../test/instances/pegasus_random/P4/CBFM-P/001_sg.txt"
ig = ising_graph(instance)
h = biases(ig)
J = couplings(ig)
Q = qubo(h, J)

model = Model(CPLEX.Optimizer)
@variable(model, x[i ∈ vertices(ig)], Bin)
@objective(model, Min, x.data' * Q * x.data)
optimize!(model)

σ = value.(x)
E = σ' * Q * σ + sum(h) - sum(J)
