using Profile, PProf
using FlameGraphs

function branch_states(r::Int, states::Matrix{Int})
    k, l = size(states)
    local_basis = collect(1:r)
    ns = Array{Int}(undef, l+1, r, k)
    ns[1:l, :, :] .= reshape(states, l, 1, k)
    ns[l, :, :] .= reshape(local_basis, r, 1, 1)
    reshape(ns, l+1, k * r)
    0
end

nstates = 1000
lstate = 128

mat_states = ones(Int, lstate, nstates)
vec_states = [mat_states[:, i]  for i in 1:nstates]

num_states = 256

branch_states(num_states, mat_states)
@profile branch_states(num_states, mat_states)

pprof(flamegraph(); webhost = "localhost", webport = 57412)
