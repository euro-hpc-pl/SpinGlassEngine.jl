using CPLEX, JuMP
using SpinGlassNetworks, LightGraphs
using LinearAlgebra

function ising2qubo(h::Vector{T}, J::Matrix{T}) where T
    n = length(h)
    Q = zeros(Float64, n, n)
    for i in 1:n
        Q[i,i] = 2 * (h[i] - sum(J[i:i, :]) - sum(J[:, i:i]))
        for j in (i+1):n
            Q[i,j] = J[i,j] * 4
        end
    end
    return Q
end


qubo(h::Vector{T}, J::Matrix{T}) where T = 4 .* J + 2 .* Diagonal((h .- sum(J)))

#instance = "$(@__DIR__)/../test/instances/pegasus_random/P4/AC3/SpinGlass/001_sg.txt"
instance = "$(@__DIR__)/../test/instances/pathological/pegasus_3_4_1.txt"
ig = ising_graph(instance)
h = biases(ig)
J = couplings(ig)
Q = ising2qubo(h, J)

model = Model(CPLEX.Optimizer)
set_silent(model)
@variable(model, x[i ∈ vertices(ig)], Bin)
@objective(model, Min, x.data' * Q * x.data)
optimize!(model)
println(raw_status(model))
σ = value.(x.data)
s = [i == 1 ? 1 : -1 for i in σ]
println(s' * J * s + s' * h)
E = objective_value(model) + sum(J) - sum(h)

