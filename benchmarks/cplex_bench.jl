using CPLEX, JuMP
using CSV, DataFrames


function ising2qubo(h::Array{Float64, 1}, J::Array{Float64, 2})
    n = length(h)
    Q = zeros(Float64, n, n)
    for i in 1:n
        Q[i,i] = h[i] * 2
        for j in (i+1):n
            Q[i,j] = J[i,j] * 4
        end
    end
    return Q
end


function get_parameters(instance::String)
    #load
    ising = DataFrame(CSV.File(instance, types = [Int, Int, Float64], header=0, comment = "#"))
    rename!(ising, :Column1 => :i, :Column2 => :j, :Column3 => :v)
    sites = unique(ising[!, :i])
    h_dict = Dict()
    J_dict = Dict()
    for row ∈ eachrow(ising)
        if row.i == row.j 
            push!(h_dict, row.i => row.v)
        else
            push!(J_dict, (row.i, row.j) => row.v)
        end
    end
    @assert length(sites) == length(h_dict)

    # now we ensure that order of elements is right
    n = length(sites)
    h = [h_dict[site] for site ∈ sites]
    J = zeros(Float64, n, n)
    for i ∈ 1:n
        for j ∈ (i+1):n
            J[i,j] = (i, j) in keys(J_dict) ?  J_dict[(i, j)] :  0
        end
    end
    h, J, sites, h_dict, J_dict
end

instance = "$(@__DIR__)/../test/instances/pegasus_random/P4/CBFM-P/001_sg.txt"

h, J, sites, h_dict, J_dict = get_parameters(instance)
Q = ising2qubo(h, J)

model = Model(CPLEX.Optimizer)
@variable(model, x[i in sites], Bin)
@objective(model, Min, x.data' * Q * x.data)
optimize!(model)
solution = Dict(site => value.(x[site]) == 1 ? 1 : -1 for site in sites)
linear = sum(h_dict[site] * solution[site] for site in sites)
quadratic = sum(J_dict[edge] * solution[edge[1]] * solution[edge[2]] for edge in keys(J_dict))
linear + quadratic
