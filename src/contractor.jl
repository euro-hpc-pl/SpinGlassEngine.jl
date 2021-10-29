export 
    MpsContractor,
    MpoLayers

abstract type AbstractContractor end

struct MpoLayers
    main::Dict
    dress::Dict
    right::Dict
end


struct MpsParameters
    bond_dimension::Int
    variational_tol::Real
    max_num_sweeps::Int
    
    MpsParameters(bd=typemax(Int), ϵ=1E-8, sw=4) = new(bd, ϵ, sw)
end


find_layout(network::PEPSNetwork{T}) where {T} = T 


struct MpsContractor <: AbstractContractor
    peps::PEPSNetwork{T} where T
    betas::Vector{Real}
    params::MpsParameters
    layers::MpoLayers

    function MpsContractor(peps, betas, params) 
        ctr = new(peps, betas, params)
        T = find_layout(peps)
        ctr.layers = MpoLayers(T, peps.ncols)  
    end
end


function MpoLayers(::Type{T}, ncols::Int) where T <: Square{EnergyGauges}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1:ncols push!(main, i => (-1//6, 0, 3//6, 4//6)) end
    for i ∈ 1:ncols - 1 push!(main, i + 1//2 => (0,)) end  

    dress = Dict(i => (3//6, 4//6) for i ∈ 1:ncols)

    for i ∈ 1:ncols push!(right, i => (-3//6, 0)) end
    for i ∈ 1:ncols - 1 push!(right, i + 1//2 => (0,)) end 

    MpoLayers(main, dress, right)
end


function MpoLayers(::Type{T}, ncols::Int) where T <: SquareStar{EnergyGauges}
    main, dress, right = Dict(), Dict(), Dict()

    for i ∈ 1//2 : 1//2 : ncols
        ii = denominator(i) == 1 ? numerator(i) : i
        push!(main, ii => (-1//6, 0, 3//6, 4//6))
        push!(dress, ii => (3//6, 4//6))
        push!(right, ii => (-3//6, 0))
    end

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
