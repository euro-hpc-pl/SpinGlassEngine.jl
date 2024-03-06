using LinearAlgebra
using CPLEX, JuMP
using SpinGlassNetworks, Graphs

function qubo(h::Vector, J::Matrix)
    b = [sum(J[i, :]) + sum(J[:, i]) for i ∈ 1:length(h)]
    4 .* J .+ 2 .* Diagonal(h .- b)
end

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
E = σ' * Q * σ + sum(J) - sum(h)
