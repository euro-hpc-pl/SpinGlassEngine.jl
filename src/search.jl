export AbstractGibbsNetwork
export low_energy_spectrum
export Solution

# abstract type AbstractGibbsNetwork end

struct Solution
    energies::Vector{Float64}
    states::Vector{Vector{Int}}
    probabilities::Vector{Float64}
    largest_discarded_probability::Float64
end

#TODO: this can probably be done better
function _branch_state(
    cfg::Vector,
    state::Vector,
    basis::Vector,
    )
    tmp = Vector{Int}[]
    for σ ∈ basis push!(tmp, vcat(state, σ)) end
    vcat(cfg, tmp)
end

# TODO: logic here can probably be done better
function _bound(probabilities::Vector{Float64}, cut::Int)
    k = length(probabilities)
    second_phase = false

    if k > cut + 1
        k = cut + 1
        second_phase = true
    end

    idx = partialsortperm(probabilities, 1:k, rev=true)

    if second_phase
        return idx[1:end-1], probabilities[last(idx)]
    else
        return idx, -Inf
    end
end

function _branch_and_bound(
    sol::Solution,
    network::AbstractGibbsNetwork,
    node::NTuple{2, Int},
    cut::Int,
    β::Real
    )

    # branch
    pdo, eng, cfg = Float64[], Float64[], Vector{Int}[]

    k = length(local_energy(network, node))

    for (p, σ, e) ∈ zip(sol.probabilities, sol.states, sol.energies)
        pdo = [pdo; p .* conditional_probability(network, σ, β)]
        eng = [eng; e .+ update_energy(network, σ)]
        cfg = _branch_state(cfg, σ, collect(1:k))
     end

    # bound
    indices, lowest_prob = _bound(pdo, cut)
    lpCut = sol.largest_discarded_probability
    lpCut < lowest_prob ? lpCut = lowest_prob : ()

    Solution(eng[indices], cfg[indices], pdo[indices], lpCut)
end

#TODO: incorporate "going back" move to improve alghoritm
function low_energy_spectrum(
    network::AbstractGibbsNetwork,
    cut::Int,
    β::Real
)
    sol = Solution([0.], [[]], [1.], -Inf)

    perm = zeros(Int, nv(network.factor_graph)) # TODO: to be removed

    #TODO: this should be replaced with the iteration over fg that is consistent with the order network
    for i ∈ 1:network.nrows, j ∈ 1:network.ncols
        v_fg = network.factor_graph.reverse_label_map[network.vertex_map((i, j))]
        perm[v_fg] = j + network.ncols * (i - 1)
        sol = _branch_and_bound(sol, network, (i, j), cut, β)
    end
    K = partialsortperm(sol.energies, 1:length(sol.energies), rev=false)

    Solution(
        sol.energies[K],
        [ σ[perm] for σ ∈ sol.states[K] ], #TODO: to be changed
        sol.probabilities[K],
        sol.largest_discarded_probability)
end
