using BenchmarkTools

# This is what we have in search.jl and it is SLOW
@inline function branch_state(local_basis::Vector{Int}, σ::Vector{T}) where T <: Number
    vcat.(Ref(σ), local_basis)
end
function branch_states_v1(num_states::Int, states::Vector{Vector{T}}) where T <: Number
    local_basis = collect(1:num_states)
    vcat(branch_state.(Ref(local_basis), states)...)
end

# This is what we want to have in search.jl but it is also SLOW (python is FAST)
function branch_states_v2(num_states::Int, states::Matrix{T}) where T <: Number
    lstate, nstates = size(states)
    L = num_states * nstates
    local_basis = collect(1:num_states)
    new_states = Array{T}(undef, lstate + 1, L)
    new_states[1:lstate, :] = repeat(states, inner=[1, num_states])
    new_states[lstate+1, :] = repeat(local_basis, outer=[nstates])
    new_states
end

# This is the same as in branch_states_v2 but array broadcasting is wrong.
function branch_states_v3(num_states::Int, states::Matrix{T}) where T <: Number
    lstate, nstates = size(states)
    local_basis = collect(1:num_states)
    ns = Array{T}(undef, lstate+1, num_states, nstates)
    ns[1:lstate, :, :] .= reshape(states, lstate, 1, nstates)
    ns[lstate+1, :, :] .= reshape(local_basis, num_states, 1, 1)
    reshape(ns, lstate+1, nstates * num_states)
end


function branch_states_v4(num_states::Int, v::Vector{Vector{T}}) where T <: Number
    states = transpose(reduce(vcat, transpose.(v)))
    lstate, nstates = size(states)
    local_basis = collect(1:num_states)
    ns = Array{T}(undef, lstate+1, num_states, nstates)
    ns[1:lstate, :, :] .= reshape(states, lstate, 1, nstates)
    ns[lstate+1, :, :] .= reshape(local_basis, num_states, 1, 1)
    reshape(ns, lstate+1, nstates * num_states)
end


T = Int #Float64

nstates = 10000
lstate = 256

mat_states = rand(T, lstate, nstates);
vec_states = [mat_states[:, i]  for i in 1:nstates];



num_states = 256

@time x = branch_states_v1(num_states, vec_states)
@time y = branch_states_v2(num_states, mat_states)
@time z = branch_states_v3(num_states, mat_states)
@time a = branch_states_v4(num_states, vec_states)


@assert x == [y[:, i] for i ∈ 1:size(y, 2)] 
@assert x == [z[:, i] for i ∈ 1:size(z, 2)]
@assert x == [a[:, i] for i ∈ 1:size(a, 2)]
loop = 100
tt = @elapsed for _ in 1:loop branch_states_v1(num_states, vec_states) end
println(tt / loop)
# tt = @elapsed for _ in 1:loop branch_states_v2(num_states, mat_states) end
# println(tt / loop)
# tt = @elapsed for _ in 1:loop branch_states_v3(num_states, mat_states) end
# println(tt / loop)
# tt = @elapsed for _ in 1:loop branch_states_v4(num_states, vec_states) end
# println(tt / loop)


