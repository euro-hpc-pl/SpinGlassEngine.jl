m = 3
n = 4
t = 3

#instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"
#instance = "$(@__DIR__)/instances/pathological/chim_3_4_3.txt"

#instance = "$(@__DIR__)/instances/4_001.txt"
instance = "$(@__DIR__)/instances/128_001.txt"

max_states = 50
to_show=8


ig = ising_graph(instance)
println(vertices(ig))

dβ = 0.01
β = 2.

D = 16 
var_ϵ = 1E-8
sweeps = 4

control = MPSControl(D, var_ϵ, sweeps, β, dβ)
ψ = MPS(ig, control)
states, lprob, _ = solve(ψ, max_states)

#show(states[1:to_show])
#println(" ")
show(energy.(states[1:to_show], Ref(ig)))
