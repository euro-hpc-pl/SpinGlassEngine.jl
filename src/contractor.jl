export 
    MpsContractor


abstract type AbstractContractor end
abstract type AbstractStrategy end

struct BasicStrategy <: AbstractStrategy end


struct MpoLayers
    main::Dict
    dress::Dict
    right::Dict
end


struct MpsParameters
    bond_dim::Int
    var_tol::Real
    sweeps::Int

    function MpsParameters(bond_dim=typemax(Int), var_to=1E-8, sweeps=4)
        new(bond_dim, var_to, sweeps)
    end
end


struct MpsContractor{T <: AbstractStrategy} <: AbstractContractor
    peps::GibbsNetwork
    betas::Vector{Real}
    params::MpsParameters
    layers::MpoLayers

    function MpsContractor{T}(peps, betas, params) where T <: AbstractStrategy
        ctr = new(peps, betas, params)
        ctr.layers = MpoLayers(T, peps.ncols)  
    end
end


function MpoLayers(::Type{BasicStrategy}, ncols::Int)
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1:ncols push!(main, i => (-1//6, 0, 3//6, 4//6)) end
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end  

    dress = Dict(i => (3//6, 4//6) for i ∈ 1:ncols)

    for i ∈ 1:ncols push!(right, i => (-3//6, 0)) end
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end 

    MpoLayers(main, dress, right)
end


function conditional_probability(crt::MpsContractor, i::Int, j::Int)
    # should call mps
end


function mps(crt::MpsContractor, beta_index::Int)
    # wola mpo
    ## mps initial guess = mps(temp, beta_index-1) if beta_index > 1 else ....
    ## or
    ## mps_initial guess = svd truncation
    ## 2site or 1site variational ?????
    ## trzeba przekazac opcje ktore wybiora jak robimy compress
end


function mpo(layer::Dict, beta::Real)
    #wola tensor(...., beta)
end


function optimize_gauges(temp::MpsContractor)
    #for beta in betas
    
    # 1) psi_bottom =  mps  ;  psi_top = mps ( :top)
    # 2) bazujac na psi_bottom i psi_top zmienia gauge
    #    sweep left and right
        
    #end
end
