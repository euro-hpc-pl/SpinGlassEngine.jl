using Test


@testset "0 degree rotation is and identity" begin
    op = vertex_map(rotation(0), 10, 5)

    @test op((1, 1)) == (1, 1)
    @test op((10, 5)) == (10, 5)
    @test op((1, 5)) == (1, 5)
    @test op((10, 1)) == (10, 1)
    @test op((4, 3)) == (4, 3)
    @test op((2, 2)) == (2, 2)
end


@testset "90 degree rotation of square lattice correctly transforms vertices" begin
    op = vertex_map(rotation(90), 10, 5)

    @test op((1, 1)) == (1, 10)
    @test op((10, 5)) == (5, 1)
    @test op((1, 5)) == (5, 10)
    @test op((10, 1)) == (1, 1)
    @test op((4, 3)) == (3, 7)
end


@testset "180 degree rotation of square lattice correctly transforms vertices" begin
    op = vertex_map(rotation(180), 10, 5)

    @test op((1, 1)) == (10, 5)
    @test op((10, 5)) == (1, 1)
    @test op((1, 5)) == (10, 1)
    @test op((10, 1)) == (1, 5)
    @test op((4, 3)) == (7, 3)
end


@testset "270 degree rotation of square lattice correctly transforms vertices" begin
    op = vertex_map(rotation(270), 10, 5)

    @test op((1, 1)) == (5, 1)
    @test op((10, 5)) == (1, 10)
    @test op((1, 5)) == (1, 1)
    @test op((10, 1)) == (5, 10)
    @test op((4, 3)) == (3, 4)
    @test op((2, 2)) == (4, 2)
end


@testset "Reflection around x axis correctly transforms vertices" begin
    op = vertex_map(reflection(:x), 10, 5)

    @test op((1, 1)) == (10, 1)
    @test op((10, 5)) == (1, 5)
    @test op((1, 5)) == (10, 5)
    @test op((10, 1)) == (1, 1)
    @test op((4, 3)) == (7, 3)
    @test op((2, 2)) == (9, 2)
end


@testset "Reflection around y axis correctly transforms vertices" begin
    op = vertex_map(reflection(:y), 10, 5)

    @test op((1, 1)) == (1, 5)
    @test op((10, 5)) == (10, 1)
    @test op((1, 5)) == (1, 1)
    @test op((10, 1)) == (10, 5)
    @test op((4, 3)) == (4, 3)
    @test op((2, 2)) == (2, 4)
end


@testset "Reflection around diag correctly transforms vertices" begin
    op = vertex_map(reflection(:diag), 10, 5)

    @test op((1, 1)) == (1, 1)
    @test op((10, 5)) == (5, 10)
    @test op((1, 5)) == (5, 1)
    @test op((10, 1)) == (1, 10)
    @test op((4, 3)) == (3, 4)
    @test op((2, 2)) == (2, 2)
end


@testset "Reflection around antydiag correctly transforms vertices" begin
    op = vertex_map(reflection(:antydiag), 10, 5)

    @test op((1, 1)) == (5, 10)
    @test op((10, 5)) == (1, 1)
    @test op((1, 5)) == (1, 10)
    @test op((10, 1)) == (5, 1)
    @test op((4, 3)) == (3, 7)
    @test op((2, 2)) == (4, 9)
end


for transform ∈ (rotation.([0, 90, 180, 270])..., reflection.([:x, :y, :diag, :antydiag])...)
@testset "$(transform) of square lattice is a bijection" begin
    op = vertex_map(transform, 3, 3)
    all_points = [(i, j) for i ∈ 1:3, j ∈ 1:3]
    @test Set(op.(all_points)) == Set(all_points)
end
end


for transform ∈ (rotation.([90, 270])..., reflection.([:diag, :antydiag])...)
@testset "$(transform) of rectangular lattice are bijection flipping lattice dimensions" begin
    op = vertex_map(transform, 2, 3)
    original_points = [(i, j) for i ∈ 1:2, j ∈ 1:3]
    transformed_points = [(i, j) for i ∈ 1:3, j ∈ 1:2]
    @test Set(op.(original_points)) == Set(transformed_points)
end
end


for transform ∈ (rotation.([0, 180])..., reflection.([:x, :y])...)
@testset "$(transform) of rectangular lattice are bijection leaving lattice dimensions unchanged" begin
    op = vertex_map(transform, 7, 3)
    all_points = [(i, j) for i ∈ 1:7, j ∈ 1:3]
    @test Set(op.(all_points)) == Set(all_points)
end
end


for transform ∈ (rotation.([0, 90, 180, 270])..., reflection.([:x, :y, :diag, :antydiag])...)
@testset "It is not possible to translate point outside of the lattice with $(transform)" begin
    op = vertex_map(transform, 4, 3)
    @test_throws ArgumentError op((20, 7))
end
end
