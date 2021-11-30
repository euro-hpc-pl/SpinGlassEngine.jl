using SpinGlassEngine
using Logging
using SpinGlassNetworks, SpinGlassTensors
using LightGraphs
using LinearAlgebra
using TensorCast
using MetaGraphs
using CSV


disable_logging(LogLevel(1))

function proj(state, dims::Union{Vector, NTuple})
    P = Matrix{Float64}[]
    for (σ, r) ∈ zip(state, dims)
        v = zeros(r)
        v[idx(σ)...] = 1.
        push!(P, v * v')
    end
    P
end

function SpinGlassEngine.tensor(ψ::AbstractMPS, state::State)
    C = I
    for (A, σ) ∈ zip(ψ, state)
        C *= A[:, idx(σ), :]
    end
    tr(C)
end

function SpinGlassEngine.tensor(ψ::MPS)
    dims = rank(ψ)
    Θ = Array{eltype(ψ)}(undef, dims)

    for σ ∈ all_states(dims)
        Θ[idx.(σ)...] = tensor(ψ, σ)
    end
    Θ
end


function bench(instance_dir::String)
    m = 16
    n = 16
    t = 8
    L = n * m * t

    δp = 1E-2
    bond_dim = 16
    num_states = 100
    max_num_sweeps = 4
    variational_tol = 1E-8
    betas = collect(1:8)

    dir = cd(instance_dir)

    open("/home/tsmierzchalski/tensor/chimera.csv", "w") do file

    for (i, instance) ∈ enumerate(readdir(join=true))
        if instance == "/home/tsmierzchalski/tensor/chimera2048_spinglass_power/groundstates_otn2d.txt" continue end
        ig = ising_graph(instance)

        fg = factor_graph(
            ig,
            #spectrum=full_spectrum,
            cluster_assignment_rule=super_square_lattice((m, n, t))
        )

        params = MpsParameters(bond_dim, variational_tol, max_num_sweeps)
        search_params = SearchParameters(num_states, δp)

        for β ∈ betas, Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense, )
            for Layout ∈ (EnergyGauges, GaugesEnergy, EngGaugesEng), transform ∈ rotation.([0])

                network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
                ctr = MpsContractor{Strategy}(network, [β], params)

                times = @elapsed sol = low_energy_spectrum(ctr, search_params, merge_branches(network))
                data = Dict(
                    "i" => i, "β" => β, "Strategy" => Strategy,
                    "Sparsity" => Sparsity, "Layout" => Layout,
                    "transform" => transform, "energies" => sol.energies[1:1], 
                    "time" => times
                )
                CSV.write(file, data, delim = ';', append = true)
            end
        end
    end
end
end

bench("/home/tsmierzchalski/tensor/chimera2048_spinglass_power")
