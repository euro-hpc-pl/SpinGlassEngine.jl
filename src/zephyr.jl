export ZephyrSquare

"""
$(TYPEDSIGNATURES)
"""
struct ZephyrSquare <: AbstractGeometry end

"""
$(TYPEDSIGNATURES)
"""
# Only Z1
function ZephyrSquare(m::Int, n::Int)
    labels = [(i, j, k) for i ∈ 1:2*n for j ∈ 1:2*m for k ∈ 1:2]  # change for bigger zephyr
    lg = LabelledGraph(labels)
    for i ∈ 1:2*n, j ∈ 1:2*m add_edge!(lg, (i, j, 1), (i, j, 2)) end

    # horizontals
    for i ∈ 1:n, j ∈ 1:m
        add_edge!(lg, (i, j, 1), (i, j+1, 2))
        add_edge!(lg, (i, j, 2), (i, j+1, 1))
        add_edge!(lg, (i+1, j, 2), (i+1, j+1, 1))
        add_edge!(lg, (i+1, j, 1), (i+1, j+1, 2))
    end

    #vertical
    for i ∈ 1:n, j ∈ 1:m
        add_edge!(lg, (i, j, 1), (i+1, j, 2))
        add_edge!(lg, (i, j, 2), (i+1, j, 1))
        add_edge!(lg, (i, j+1, 2), (i+1, j+1, 1))
        add_edge!(lg, (i, j+1, 1), (i+1, j+1, 2))

    end

    # diagonals
     for i ∈ 1:m, j ∈ 1:n
         add_edge!(lg, (i, j, 2), (i+1, j+1, 2))
         add_edge!(lg, (i+1, j, 1), (i, j+1, 1))
     end
    lg
end