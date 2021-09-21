function _clear_cache(network::AbstractGibbsNetwork, sol::Solution)
    i, j = node_from_index(network, length(first(sol.states))+1)
    if j != network.ncols return end
    delete!(memoize_cache(mps), (network, i))
    delete!(memoize_cache(dressed_mps), (network, i))
    delete!(memoize_cache(mpo), (network, i-1))
    delete!(memoize_cache(_mpo), (network, i-1))
    lec = memoize_cache(left_env)
    delete!.(Ref(lec), filter(k->k[2]==i, keys(lec)))
    rec = memoize_cache(right_env)
    delete!.(Ref(rec), filter(k->k[2]==i, keys(rec)))
end