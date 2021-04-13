export LatticeTransformation, rotation, reflection


struct LatticeTransformation
    permutation::NTuple{4, Int}
end

Base.:(∘)(op1::LatticeTransformation, op2::LatticeTransformation) =
    LatticeTransformation(op1.permutation[collect(op2.permutation)])


function reflection(axis::Symbol)
    if axis == :x
        perm = (4, 3, 2, 1)
    elseif axis == :y
        perm = (2, 1, 4, 3)
    elseif axis == :diag
        perm = (1, 4, 3, 2)
    elseif axis == :antydiag
        perm = (3, 2, 1, 4)
    else
        throw(ArgumentError("Unknown reflection axis: $(axis)"))
    end
    LatticeTransformation(perm)
end


function rotation(θ::Int)
    if θ % 90 != 0
        ArgumentError("Only integral multiplicities of 90° can be passed as θ.")
    end
    θ = θ % 360
    if θ == 0
        LatticeTransformation((1, 2, 3, 4))
    elseif θ == 90
        LatticeTransformation((4, 1, 2, 3))
    else
        rotation(θ - 90) ∘ rotation(90)
    end
end


function check_bounds(m, n)
    function _check(i, j)
        if i < 1 || i > m || j < 1 || j > n
            throw(ArgumentError("Point ($(i), $(j)) not in $(m)x$(n) lattice."))
        end
        true
    end
end


function vertex_map(vert_permutation::NTuple{4, Int}, m, n)
    checker = check_bounds(m, n)
    if vert_permutation == (1, 2, 3, 4) #
        f = (i, j) -> (i, j)
    elseif vert_permutation == (4, 1, 2, 3) # 90 deg rotation
        f = (i, j) -> (j, m - i + 1)
    elseif vert_permutation == (3, 4, 1, 2) # 180 deg rotation
        f = (i, j) -> (m - i + 1, n - j + 1)
    elseif vert_permutation == (2, 3, 4, 1) # 270 deg rotation
        f = (i, j) -> (n - j + 1, i)
    elseif vert_permutation == (2, 1, 4, 3) # :y reflection
        f = (i, j) -> (i, n - j + 1)
    elseif vert_permutation == (4, 3, 2, 1) # :x reflection
        f = (i, j) -> (m - i + 1, j)
    elseif vert_permutation == (1, 4, 3, 2) # :diag reflection
        f = (i, j) -> (j, i)
    elseif vert_permutation == (3, 2, 1, 4) # :antydiag reflection
        f = (i, j) -> (n -j + 1, m - i + 1)
    else
        ArgumentError("$(vert_permutation) does not define square isometry.")
    end
    (tuple) -> checker(tuple[1], tuple[2]) && f(tuple[1], tuple[2])
end

vertex_map(trans::LatticeTransformation, m::Int, n::Int) = vertex_map(trans.permutation, m, n)
