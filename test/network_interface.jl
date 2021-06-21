@testset "fuse_projectors correctly fuses projectors - 2x2 matrix." begin
    projectors = ([1 1; 0 1], [1 0; 1 1])
    expected_fused = [1 0; 0 1]
    expected_transitions = [[1 1; 0 1], [1 0; 1 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end

@testset "fuse_projectors correctly fuses projectors - another 2x2 matrix." begin
    projectors = ([1 1; 0 0], [1 0; 0 1])
    expected_fused = [1 0; 0 1]
    expected_transitions = [[1 1; 0 0], [1 0; 0 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end

@testset "fuse_projectors correctly fuses projectors - 3x3 matrix." begin
    projectors = ([1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1])
    expected_fused = [1 0 0; 0 1 0; 0 0 1]
    expected_transitions = [[1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end

@testset "fuse_projectors correctly fuses projectors - another 3x3 matrix." begin
    projectors = ([1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1])
    expected_fused = [1 0 0; 0 1 0; 0 0 1]
    expected_transitions = [[1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end