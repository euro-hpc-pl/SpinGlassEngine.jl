
using LabelledGraphs

function prune(ig::IsingGraph) 
    idx = findall(!iszero, degree(ig))
    gg = ig[ig.labels[idx]]
    labels = collect(vertices(gg.inner_graph))
    reverse_label_map = Dict(i => i for i=1:nv(gg.inner_graph))
    LabelledGraph(labels, gg.inner_graph, reverse_label_map)
end

@testset "MPS based search finds the correct low energy spectrum" begin

    instance = "$(@__DIR__)/instances/pathological/cross_3_4_dd.txt"

    ig = ising_graph(instance)
    ig = prune(ig) 

    expected_energies = [-23.301855, -23.221513, -23.002799, -22.922457]

    max_states = 100
    to_show = length(expected_energies)

    dβ = 2.0
    β = 2.0
    
    Dcut = 32
    var_ϵ = 1E-8
    max_sweeps = 4

    @testset "without purifications" begin
        schedule = fill(dβ, Int(ceil(β/dβ)))
        ψ = MPS(ig, Dcut, var_ϵ, max_sweeps, schedule)
        println(dot(ψ, ψ))
        states, lprob, _ = solve(ψ, max_states)
        println(dot(ψ, ψ))
        @test energy.(states[1:to_show], Ref(ig)) ≈ expected_energies
    end

    #@testset "using purifications" begin
    #    ψ = MPS(ig, Dcut, var_ϵ, max_sweeps, β, dβ, :lin)
    #    states, lprob, _ = solve(ψ, max_states)
    #    @test energy.(states[1:to_show], Ref(ig)) ≈ expected_energies
    #end
end 
