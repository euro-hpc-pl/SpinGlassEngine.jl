m = 3
n = 4
t = 3
N = m * n * t 
instance = "$(@__DIR__)/instances/pathological/cross_$(m)_$(n)_dd.txt"

#L = 2
#N = L^2
#instance = "$(@__DIR__)/instances/$(N)_001.txt"

ig = ising_graph(instance)
dβ = 0.01
β = 1.


r = fill(2, N)
set_prop!(ig, :rank, r)

ϵ = 1E-8
D = prod(r) + 1
var_ϵ = 1E-8
sweeps = 4
max_states = 50

control = MPSControl(D, var_ϵ, sweeps, β, dβ)
states, lprob, _ = solve(MPS(ig, control), max_states)

to_show=10
show(states[1:to_show])
println(" ")
show(lprob)
println(" ")
show(energy.(states[1:to_show], Ref(ig)))
