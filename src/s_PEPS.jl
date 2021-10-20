export 
    PEPSNetwork, 
    node_from_index, 
    conditional_probability


struct AbstractTensors end

#TODO : organize this into structures

function peps_lattice(m::Int, n::Int)
    labels = [(i, j) for j ∈ 1:n for i ∈ 1:m]
    LabelledGraph(labels, grid((m, n)))
end


#=
struct SquareGeometry
    nrows::Int
    ncols::Int
    map::Dict

    function SquareGeometry(n::Int, m::Int, map::Dict=Dict())
        ct = new(nrows, ncols)
        for i ∈ 1:nrows, j ∈ 1:ncols
            push!(ct.map, (i, j) => :site)
            # why if?
            if j < ncols push!(ct.map, (i, j + 1//2) => :central_h) end
            if i < nrows push!(ct.map, (i + 1//2, j) => :central_v) end
        end
    end
end
=#
#=
struct SquareGeometry_sparse
    nrows::Int
    ncols::Int
    map::Dict

    function SquareGeometry(n::Int, m::Int, map::Dict=Dict())
        ct = new(nrows, ncols)
        for i ∈ 1:nrows, j ∈ 1:ncols
            push!(ct.map, (i, j) => :site_sparse)
            if j < ncols push!(ct.map, (i, j + 1//2) => :central_h) end
            if i < nrows push!(ct.map, (i + 1//2, j) => :central_v) end
        end
    end
end
=#
#=
struct Chimera_contraction_strategy_no_1  #zwezanie przy pomocy boundary mps
    ncols::Int
    main::Dict
    dress::Dict
    right::Dict

    function MpoLayers(
        ncols::Int,
        main::Dict=Dict(),
        dress::Dict=Dict(),
        right::Dict=Dict()
    )
        ml = new(ncols)

        for i ∈ 1:ncols push!(ml.main, i => (-1//6, 0, 3//6, 4//6)) end
        for i ∈ 1:ncols - 1 push!(ml.main, i + 1//2 => (0,)) end  

        ml.dress = Dict(i => (3//6, 4//6) for i ∈ 1:ncols)

        for i ∈ 1:ncols push!(ml.right, i => (-3//6, 0)) end
        for i ∈ 1:ncols - 1 push!(ml.right, i + 1//2 => (0,)) end  
    end
end

dla square lattice
for i ∈ 1:nrows-1, j ∈ 1:ncols
            push!(ct.map, (i + 4//6, j) => :gauge_h)
            push!(ct.map, (i + 5//6, j) => :gauge_h)
        end

        for i ∈ 1:nrows-1, j ∈ 1:ncols
            push!(ct.map, (i + 1//6, j) => :gauge_h)
            push!(ct.map, (i + 2//6, j) => :gauge_h)
        end
=#


#=  
struct Chimera_contraction_strategy_no_2   #zwezanie przy pomocy boundary mps
    ncols::Int
    main::Dict
    dress::Dict
    right::Dict

    function MpoLayers(
        ncols::Int, 
        main::Dict=Dict(), 
        dress::Dict=Dict(), 
        right::Dict=Dict()
    )
        ml = new(ncols)

        for i ∈ 1:ncols push!(ml.main, i => (-1//6, 0, 3//6, 4//6)) end
        for i ∈ 1:ncols - 1 push!(ml.main, i + 1//2 => (0,)) end  

        ml.dress = Dict(i => (3//6, 4//6) for i ∈ 1:ncols)

        for i ∈ 1:ncols push!(ml.right, i => (-3//6, 0)) end
        for i ∈ 1:ncols - 1 push!(ml.right, i + 1//2 => (0,)) end  
    end
end

dla square lattice
for i ∈ 1:nrows-1, j ∈ 1:ncols
            push!(ct.map, (i + 1//6, j) => :gauge_h)
            push!(ct.map, (i + 2//6, j) => :gauge_h)
        end

        for i ∈ 1:nrows-1, j ∈ 1:ncols
            push!(ct.map, (i + 1//6, j) => :gauge_h)
            push!(ct.map, (i + 2//6, j) => :gauge_h)
        end
=#



#=
struct DiagonalGeometry{sparsity::Bool}
    nrows::Int
    ncols::Int
    map::Dict

    function DiagonalGeometry{sparsity}(n::Int, m::Int, map::Dict=Dict())
        ct = new(nrows, ncols)
        site_type = sparsity ? :site_sparse : :site
        virtual_type = sparsity ? :virtual_sparse : :virtual

        for i ∈ 1:nrows, j ∈ 1:ncols
            push!(ct.map, (i, j) => site_type)
            push!(ct.map, (i, j - 1//2) => :virtual_type)
            push!(ct.map, (i + 1//2, j) => :central_v)
        end
        for i ∈ 1:nrows-1, j ∈ 0:ncols-1
            push!(ct.map, (i + 1//2, j + 1//2) => :central_d)
        end
    end
end
=#



#=
 for i ∈ 1 : nrows - 1, j ∈ 1//2 : 1//2 : ncols
        jj = denominator(j) == 1 ? numerator(j) : j
        push!(_tensors_map, (i + 4//6, jj) => :gauge_h)
        push!(_tensors_map, (i + 5//6, jj) => :gauge_h)
    end
=#
#=
struct SquareGeometry
    nrows::Int
    ncols::Int
    map::Dict

    function SquareGeometry(n::Int, m::Int, map::Dict=Dict())
        ct = new(nrows, ncols)
        for i ∈ 1:nrows, j ∈ 1:ncols
            push!(ct.map, (i, j) => :site)
            push!(ct.map, (i, j + 1//2) => :central_h)
            push!(ct.map, (i, j + 1//2) => :central_v)
        end
        for i ∈ 1:nrows-1, j ∈ 1:ncols
            push!(ct.map, (i + 4//6, j) => :gauge_h)
            push!(ct.map, (i + 5//6, j) => :gauge_h)
        end
    end
end





struct Contraction
    bond_dim::Int
    var_tol::Real
    sweeps::Int
end


struct _PEPSNetwork{network_layout} 
    # ROBI: pozwala wygenerowac tensor ze wzgledu na wspolrzedne w tensor_map i podany parametr beta
    factor_graph::LabelledGraph{T, NTuple{2, Int}} where T
    network_graph::LabelledGraph{S, NTuple{2, Int}} where S
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    tensors_map::AbstractTensors
    gauges::Dict

    function _PEPSNetwork{network_layout}(
        m::Int,
        n::Int,
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation;
        network_layout #typ sieci (chimera_v1 chimera_v2_, chimera_v3, diagonal_v1, etc.)
    )
        vmap = vertex_map(transformation, m, n)
        network_graph = peps_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        if !is_compatible(factor_graph, network_graph)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        net = new(factor_graph, network_graph, vmap, m, n, nrows, ncols)
        net.tensors_map = setup_layout(network_layout, nrows, ncols)  # network_layout sluzy do wygenerowania konkretnego tensor map
        initialize_gauges(net, :id)

        net
    end
end

# moze sie zmienic gauge
# β::Real jako parametr w generacji tensora

struct MpsContractor <: AbstractContractor
    peps::_PEPSNetwork
    MpoLayers
    betas::Real
    contraction_scheme # -- struktura ktora nam powie jak ograc compress
end

function conditional_probability(temp::MpsContractor, ii, jj)
    #wola mps
end


function mps(temp::MpsContractor, beta_index::Int)
    # wola mpo
    ## mps initial guess = mps(temp, beta_index-1) if beta_index > 1 else ....
    ## or
    ## mps_initial guess = svd truncation
    ## 2site or 1site variational ?????
    ## trzeba przekazac opcje ktore wybiora jak robimy compress
end

function mpo(layer::dict, beta::Real)
    #wola tensor(...., beta)
end


function optimize_gauges(temp::MpsContractor)
    #for beta in betas
    
    # 1) psi_bottom =  mps  ;  psi_top = mps ( :top)
    # 2) bazujac na psi_bottom i psi_top zmienia gauge
    #    sweep left and right
        

    #end
end
=#
#=
struct _PEPSNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
    factor_graph::LabelledGraph{T, NTuple{2, Int}} where T
    network_graph::LabelledGraph{S, NTuple{2, Int}} where S
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    β::Real
    contraction_parmas::Contraction
    tensors_map::ChimeraTensors
    mpo_layers::MpoLayers
    gauges::Dict

    function _PEPSNetwork(
        m::Int,
        n::Int,
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation;
        β::Real,
        contraction_parmas::Contraction=Contraction(typemax(Int), 1E-8, 4),
        gauges::Dict=Dict() 
    )
        vmap = vertex_map(transformation, m, n)
        network_graph = peps_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        if !is_compatible(factor_graph, network_graph)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        net = new(factor_graph, network_graph, vmap, m, n, nrows, ncols, β, gauges)

        net.contraction_parmas = contraction_parmas
        net.tensors_map = ChimeraTensors(nrows, ncols)
        net.mpo_layers = MpoLayers(ncols)
        update_gauges!(net, :id)

        net
    end
end
=#

struct PEPSNetwork <: AbstractGibbsNetwork{NTuple{2, Int}, NTuple{2, Int}}
    factor_graph::LabelledGraph{T, NTuple{2, Int}} where T
    network_graph::LabelledGraph{S, NTuple{2, Int}} where S
    vertex_map::Function
    m::Int
    n::Int
    nrows::Int
    ncols::Int
    β::Real
    #
    bond_dim::Int
    var_tol::Real
    sweeps::Int
    #
    gauges
    tensors_map
    #
    mpo_main::Dict
    mpo_dress::Dict
    mpo_right::Dict

    function PEPSNetwork(
        m::Int,
        n::Int,
        factor_graph::LabelledGraph,
        transformation::LatticeTransformation;
        β::Real,
        bond_dim::Int=typemax(Int),
        var_tol::Real=1E-8,
        sweeps::Int=4,
    )
        vmap = vertex_map(transformation, m, n)
        ng = peps_lattice(m, n)
        nrows, ncols = transformation.flips_dimensions ? (n, m) : (m, n)

        if !is_compatible(factor_graph, ng)
            throw(ArgumentError("Factor graph not compatible with given network."))
        end

        _tensors_map = Dict()
        for i ∈ 1:nrows, j ∈ 1:ncols
            push!(_tensors_map, (i, j) => :site)
            push!(_tensors_map, (i, j + 1//2) => :central_h)
            push!(_tensors_map, (i + 1//2, j) => :central_v)
        end
        for i ∈ 1:nrows-1, j ∈ 1:ncols
            push!(_tensors_map, (i + 4//6, j) => :gauge_h)
            push!(_tensors_map, (i + 5//6, j) => :gauge_h)
        end

        _mpo_main = Dict()
        for i ∈ 1:ncols push!(_mpo_main, i => (-1//6, 0, 3//6, 4//6)) end
        for i ∈ 1:ncols - 1 push!(_mpo_main, i + 1//2 => (0,)) end  # consier changing (0,) to 0

        # MPO : Dict(Q => tensor, Q => lista tensorow)
        # gdzie sie da to w 2gim przypadku iteracja po liscie da dispatch do operacji na pojedynczym tensor_size

        _mpo_dress = Dict(i => (3//6, 4//6) for i ∈ 1:ncols)

        _mpo_right = Dict()
        for i ∈ 1:ncols push!(_mpo_right, i => (-3//6, 0)) end
        for i ∈ 1:ncols - 1 push!(_mpo_right, i + 1//2 => (0,)) end  # consier changing (0,) to 0

        _gauges = Dict()

        network = new(factor_graph, ng, vmap, m, n, nrows, ncols, β, bond_dim,
                      var_tol, sweeps, _gauges, _tensors_map,
                      _mpo_main, _mpo_dress, _mpo_right
                )
        update_gauges!(network, :id)
        network
    end
end


function projectors(network::PEPSNetwork, vertex::NTuple{2, Int})
    i, j = vertex
    neighbours = ((i, j-1), (i-1, j), (i, j+1), (i+1, j))
    projector.(Ref(network), Ref(vertex), neighbours)
end


node_index(peps::AbstractGibbsNetwork, node::NTuple{2, Int}) = peps.ncols * (node[1] - 1) + node[2]
_mod_wo_zero(k, m) = k % m == 0 ? m : k % m


node_from_index(peps::AbstractGibbsNetwork, index::Int) =
    ((index-1) ÷ peps.ncols + 1, _mod_wo_zero(index, peps.ncols))


function boundary(peps::PEPSNetwork, node::NTuple{2, Int})
    i, j = node
    vcat(
        [
            ((i, k), (i+1, k)) for k ∈ 1:j-1
        ]...,
            ((i, j-1), (i, j)),
        [
            ((i-1, k), (i, k)) for k ∈ j:peps.ncols
        ]...
    )
end


function conditional_probability(peps::PEPSNetwork, w::Vector{Int})
    i, j = node_from_index(peps, length(w)+1)
    ∂v = boundary_state(peps, w, (i, j))

    L = left_env(peps, i, ∂v[1:j-1])
    R = right_env(peps, i, ∂v[j+2 : peps.ncols+1])
    A = reduced_site_tensor(peps, (i, j), ∂v[j], ∂v[j+1])

    ψ = dressed_mps(peps, i)
    M = ψ.tensors[j]

    @tensor prob[σ] := L[x] * M[x, d, y] * A[r, d, σ] *
                       R[y, r] order = (x, d, r, y)

    normalize_probability(prob)
end


function bond_energy(
    network::AbstractGibbsNetwork, 
    u::NTuple{2, Int}, 
    v::NTuple{2, Int}, 
    σ::Int
)
    fg_u, fg_v = network.vertex_map(u), network.vertex_map(v)
    if has_edge(network.factor_graph, fg_u, fg_v)
        pu, en, pv = get_prop.(Ref(network.factor_graph), Ref(fg_u), Ref(fg_v), (:pl, :en, :pr))
        energies = en[pu, pv[σ:σ, :]]
    elseif has_edge(network.factor_graph, fg_v, fg_u)
        pv, en, pu = get_prop.(Ref(network.factor_graph), Ref(fg_v), Ref(fg_u), (:pl, :en, :pr))
        energies = en[pv[σ:σ, :], pu]
    else
        energies = zeros(length(local_energy(network, u)))
    end
    vec(energies)
end


function update_energy(network::PEPSNetwork, σ::Vector{Int})
    i, j = node_from_index(network, length(σ)+1)
    bond_energy(network, (i, j), (i, j-1), local_state_for_node(network, σ, (i, j-1))) +
    bond_energy(network, (i, j), (i-1, j), local_state_for_node(network, σ, (i-1, j))) +
    local_energy(network, (i, j))
end
