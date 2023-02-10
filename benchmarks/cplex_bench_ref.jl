using LinearAlgebra
using CPLEX, JuMP
using SpinGlassNetworks, LightGraphs

qubo(h::Vector, J::Matrix) = 4 .* J .+ 2 .* Diagonal(h .- sum(J))

#instance = "$(@__DIR__)/../test/instances/pegasus_random/P4/CBFM-P/001_sg.txt"
instance = Dict((1, 2) => 1.0, (1, 3) => 1.0)
ig = ising_graph(instance)
h = biases(ig)
J = couplings(ig)
Q = qubo(h, J)

model = Model(CPLEX.Optimizer)
@variable(model, x[i ∈ vertices(ig)], Bin)
@objective(model, Min, x.data' * Q * x.data)
optimize!(model)

σ = value.(x)
E = σ' * Q * σ + sum(J) - sum(h)
println(σ)
println(E)

sp = brute_force(ig, :CPU; num_states=1)
en = sp.energies[1]
st = sp.states[1]
σ2 = (st .+ 1) ./ 2
println(σ2)
println(en)
@assert en ≈ σ2' * Q * σ2 + sum(J) - sum(h)
