# This file should be used to quickly calculate spectrum for testing instances
# WARNING: There is a problem  with solve when max_states is larger enough 1

m = 3
n = 4
t = 3

#instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"

L = 2
N = L^2

instance = "$(@__DIR__)/instances/$(N)_001.txt"

ig = ising_graph(instance)
dβ = 0.01
β = 1.

#N = m * n * t 
r = fill(2, N)
set_prop!(ig, :rank, r)

ϵ = 1E-8
D = prod(r) + 1
var_ϵ = 1E-8
sweeps = 4
max_states = 8

control = MPSControl(D, var_ϵ, sweeps, β, dβ)
Gψ = MPS(ig, control)

states, lprob, _ = solve(Gψ, max_states)
println(lprob)
println(size(states))
energie = [energy(states[i, :], ig) for i ∈ 1:size(states, 1)]
#energie = [energy(σ, ig) for σ ∈ eachrow(states)] # This should work!
println(energie)
