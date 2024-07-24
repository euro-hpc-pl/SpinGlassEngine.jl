@testset "Example image is correcly loaded and computed" begin
    instance_dir = "$(@__DIR__)/instances/rmf/n4/penguin-small.h5"
    onGPU = true
    β = 1.0
    bond_dim = 8
    δp = 1E-4
    num_states = 64
    cl_h = potts_hamiltonian(instance_dir)
    Nx, Ny = get_prop(cl_h, :Nx), get_prop(cl_h, :Ny)
    params = MpsParameters{Float64}(; bd = bond_dim, ϵ = 1E-8, sw = 4)
    search_params = SearchParameters(; max_states = num_states, cut_off_prob = δp)
    Gauge = NoUpdate
    graduate_truncation = :graduate_truncate
    energies = Vector{Float64}[]
    Strategy = Zipper
    Layout = GaugesEnergy
    Sparsity = Sparse
    transform = rotation(0)
    net = PEPSNetwork{KingSingleNode{Layout},Sparsity}(Nx, Ny, cl_h, transform)
    ctr = MpsContractor{Strategy,Gauge,Float64}(
        net,
        params;
        onGPU = onGPU,
        βs = [β / 8, β / 4, β / 2, β],
        graduate_truncation = :graduate_truncate,
        mode = :RMF,
    )
    sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
end
