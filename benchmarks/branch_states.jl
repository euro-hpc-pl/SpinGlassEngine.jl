
@inline function to_vec(mat::Matrix{T}) where T <: Number
    [mat[:, i] for i ∈ 1:size(mat, 2)]
end

@inline function to_mat(vec::Vector{Vector{T}}) where T <: Number
    transpose(reduce(vcat, transpose.(vec)))
end
@inline function branch_state(local_basis::Vector{Int}, σ::Vector{T}) where T <: Number
    vcat.(Ref(σ), local_basis)
end
function branch_states_v1(num_states::Int, states::Vector{Vector{T}}) where T <: Number
    local_basis = collect(1:num_states)
    vcat(branch_state.(Ref(local_basis), states)...)
end

function branch_states_v2(num_states::Int, states::Matrix{T}) where T <: Number
    lstate, nstates = size(states)
    L = num_states * nstates
    local_basis = collect(1:num_states)
    new_states = Matrix{T}(undef, lstate + 1, L)
    new_states[1:lstate, :] = repeat(states, inner=[1, num_states])
    new_states[lstate+1, :] = repeat(local_basis, outer=[nstates])
    new_states
end

function branch_states_v3(num_states::Int, states::Matrix{T}) where T <: Number
    lstate, nstates = size(states)
    local_basis = collect(1:num_states)
    ns = Array{T}(undef, lstate+1, num_states, nstates)
    ns[1:lstate, :, :] .= reshape(states, lstate, 1, nstates)
    ns[lstate+1, :, :] .= reshape(local_basis, num_states, 1, 1)
    reshape(ns, lstate+1, nstates * num_states)
end

function branch_states_v4(num_states::Int, vec_states::Vector{Vector{T}}) where T <: Number
    states = to_mat(vec_states)
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

mat_states = rand(T, lstate, nstates)
vec_states = to_vec(mat_states)

num_states = 256

@time x = branch_states_v1(num_states, vec_states)
@time y = branch_states_v2(num_states, mat_states)
@time z = branch_states_v3(num_states, mat_states)
@time w = branch_states_v4(num_states, vec_states)

@assert x == to_vec(y) == to_vec(z) == to_vec(w)

println()

loop = 10
for foo ∈ (branch_states_v1, branch_states_v4)
    t = @elapsed for _ ∈ 1:loop foo(num_states, vec_states) end
    println("time for $(foo) ", t / loop)
end

for foo ∈ (branch_states_v2, branch_states_v3)
    t2 = @elapsed for _ ∈ 1:loop foo(num_states, mat_states) end
    println("time for $(foo) ", t2 / loop)
end

t3 = @elapsed for _ ∈ 1:loop to_mat(vec_states) end
println("time for to_mat ", t3 / loop)
