[![Coverage Status](https://coveralls.io/repos/github/iitis/SpinGlassEngine.jl/badge.svg?branch=master)](https://coveralls.io/github/iitis/SpinGlassEngine.jl?branch=master)
# SpinGlassEngine.jl

## Motivation

Package containing the core algorithms for solving Ising instances with tensor networks.


## Usage

An example of a simple use case - solving an Ising instance on the Chimera graph along with a benchmark.

```julia
using SpinGlassNetworks
using SpinGlassTensors
using SpinGlassEngine

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

    @time fg = factor_graph(
        ising_graph(instance),
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
    )

    params = MpsParameters(bond_dim, 1E-8, 10)
    search_params = SearchParameters(num_states, δp)

    for Strategy ∈ (SVDTruncate, ), Sparsity ∈ (Dense,)
        for Layout ∈ (EnergyGauges, ), transform ∈ rotation.([0])
            println((Strategy, Sparsity, Layout, transform))

            @time network = PEPSNetwork{Square{Layout}, Sparsity}(m, n, fg, transform)
            @time ctr = MpsContractor{Strategy}(network, [β/8, β/4, β/2, β], params)

            @time sol = low_energy_spectrum(ctr, search_params, merge_branches(network))

            @assert sol.energies[begin] ≈ ground_energy

            clear_memoize_cache()
        end
    end
end

bench("$(@__DIR__)/../test/instances/chimera_droplets/2048power/001.txt")
```

The key parts of this code are

- creation of a `factor graph` utilized in the tensor network contraction
```julia
factor_graph(
        ising_graph(instance), #read the instance as a graph
        max_cl_states,
        spectrum=brute_force,
        cluster_assignment_rule=super_square_lattice((m, n, t))
)
```

- definition of the parameters of the tensor network (dimensions, tolerances, ...)
```julia
params = MpsParameters(bond_dim, 1E-8, 10)
```

- definition of the parameters for network contraction algorithms
```julia
search_params = SearchParameters(num_states, δp)
```

- finally, approximation of the low energy spectrum
```julia
sol = low_energy_spectrum(ctr, search_params, merge_branches(network))
```
