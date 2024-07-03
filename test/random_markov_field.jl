@testset "Example image is correcly loaded and computed" begin
    instance_dir = "$(@__DIR__)/instances/rmf/n4/penguin-small.h5"
    onGPU = true
    β = 1.0
    bond_dim = 8
    δp = 1E-4
    num_states = 64
    cl_h = clustered_hamiltonian(instance_dir)
    Nx, Ny = get_prop(cl_h, :Nx), get_prop(cl_h, :Ny)
    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)
    Gauge = NoUpdate
    graduate_truncation = :graduate_truncate
    energies = Vector{Float64}[]
    Strategy = Zipper
    Layout = GaugesEnergy
    Sparsity = Sparse
    transform = rotation(0)
    net = PEPSNetwork{SquareCrossSingleNode{Layout},Sparsity}(Nx, Ny, cl_h, transform)
    ctr = MpsContractor{Strategy,Gauge}(
        net,
        [β / 8, β / 4, β / 2, β],
        graduate_truncation,
        params;
        onGPU = onGPU,
        mode = :RMF,
    )
    sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr))
    # sol_peps, s = low_energy_spectrum(ctr, search_params, merge_branches(ctr, :nofit, SingleLayerDroplets(100.0, 100, :hamming, :RMF)))
end
