using BenchmarkTools

@inline branch_state(local_basis::Vector{Int}, σ::Vector{Int}) = vcat.(Ref(σ), local_basis)
function branch_states_v1(num_states::Int, states::Vector{Vector{Int}})
    local_basis = collect(1:num_states)
    vcat(branch_state.(Ref(local_basis), states)...)
end

function branch_states_v2(num_states::Int, states::Matrix{Int})
    lstate, nstates = size(states)
    L = num_states * nstates
    local_basis = collect(1:num_states)
    new_states = Array{Int}(undef, lstate + 1, L)
    new_states[1:lstate, :] = repeat(states, outer=[1, num_states])
    new_states[lstate+1, :] = repeat(local_basis, outer=[nstates])
    new_states
end

function branch_states_v3(num_states::Int, states::Matrix{Int})
    lstate, nstates = size(states)
    local_basis = collect(1:num_states)
    ns = Array{Int}(undef, lstate+1, num_states, nstates)
    ns[1:lstate, :, :] .= reshape(states, lstate, 1, nstates)
    ns[lstate, :, :] .= reshape(local_basis, num_states, 1, 1)
    reshape(ns, lstate+1, nstates * num_states)
end

nstates = 1000
lstate = 128

mat_states = ones(Int, lstate, nstates);
vec_states = [mat_states[:, i]  for i in 1:nstates];

num_states = 256

@time x = branch_states_v1(num_states, vec_states)
@time y = branch_states_v2(num_states, mat_states)
@time z = branch_states_v3(num_states, mat_states)

@assert x == [y[:, i] for i ∈ 1:size(y, 2)] # == [z[:, i] for i ∈ 1:size(z, 2)]
