@testset "fuse_projectors correctly fuses projectors." begin
    projectors = ([1 1; 0 1], [1 0; 1 1])
    expected_fused = [1 0; 0 1]
    expected_transitions = [[1 1; 0 1], [1 0; 1 1]]
    fused, transitions = fuse_projectors(projectors)
    @test expected_fused == fused
    @test expected_transitions == values(transitions)
end
