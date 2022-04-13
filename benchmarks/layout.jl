using BenchmarkTools

branch_state(local_basis::Vector{Int}, σ::Vector{Int}) = vcat.(Ref(σ), local_basis)

function branch_states_v1(num_states::Int, states::Vector{Vector{Int}})
    local_basis = collect(1:num_states)
    vcat(branch_state.(Ref(local_basis), states)...)
end

function branch_states_v2(num_states::Int, states::Matrix{Int})
    local_basis = collect(1:num_states)
    (lstate, nstates) = size(states)
    # new_states = Matrix{Int}(undef, lstate + 1, num_states * nstates)
    # @inbounds new_states[1 : lstate, :] = repeat(reshape(states, lstate, nstates), outer=[1, num_states])
    # @inbounds new_states[lstate + 1, :] = repeat(local_basis, outer=[nstates])
    
    repeat(states, outer=[1, num_states])
    repeat(local_basis, outer=[nstates])
    [1]
end


nstates = 10000
lstate = 128

mat_states = ones(Int, lstate, nstates)
@time vec_states = [mat_states[:, i]  for i in 1:nstates]
println(typeof(vec_states))
println(size(vec_states))
println(size(mat_states))


num_states = 256


@benchmark branch_states_v1(num_states, vec_states);
#@benchmark branch_states_v2(num_states, mat_states);

# println(size(out))