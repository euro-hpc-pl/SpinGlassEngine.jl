using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine
using Logging
using Profile, PProf #ProfileVega

disable_logging(LogLevel(1))

function bench(instance::String)
    m = 16
    n = 16
    t = 8

    L = n * m * t
    max_cl_states = 2^(t-0)

    ground_energy = -3336.773383

    β = 3.0
    bond_dim = 32
    δp = 1E-3
    num_states = 1000

    fg = factor_graph(
        ising_graph(instance),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    net = PEPSNetwork{Square{EnergyGauges}, Dense}(m, n, fg, rotation(0))
    ctr = MpsContractor{SVDTruncate}(net, [β/8, β/4, β/2, β], params)
    sol = low_energy_spectrum(ctr, search_params, merge_branches(net))

    @assert sol.energies[begin] ≈ ground_energy
    clear_memoize_cache()
end

instance = "$(@__DIR__)/../test/instances/chimera_droplets/2048power/001.txt"
#bench(instance)

@profile bench(instance)
pprof()

#ProfileVega.view() |> save("$(@__DIR__)/prof_full_chimera.svg")
#ProfileSVG.save("$(@__DIR__)/prof_full_chimera.svg")
