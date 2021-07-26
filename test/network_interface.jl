@testset "fuse_projectors correctly fuses projectors." begin
    projectors1 = ([1 1; 0 1], [1 0; 1 1])
    expected_fused1 = [1 0; 0 1]
    expected_transitions1 = [[1 1; 0 1], [1 0; 1 1]]
    fused1, transitions1 = fuse_projectors(projectors1)
    @test expected_fused1 == fused1
    @test expected_transitions1 == values(transitions1)

    projectors2 = ([1 1; 0 0], [1 0; 0 1])
    expected_fused2 = [1 0; 0 1]
    expected_transitions2 = [[1 1; 0 0], [1 0; 0 1]]
    fused2, transitions2 = fuse_projectors(projectors2)
    @test expected_fused2 == fused2
    @test expected_transitions2 == values(transitions2)

    projectors3 = ([1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1])
    expected_fused3 = [1 0 0; 0 1 0; 0 0 1]
    expected_transitions3 = [[1 1 1; 0 0 0; 1 1 1], [1 0 0; 0 1 0; 0 0 1]]
    fused3, transitions3 = fuse_projectors(projectors3)
    @test expected_fused3 == fused3
    @test expected_transitions3 == values(transitions3)

    projectors4 = ([1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1])
    expected_fused4 = [1 0 0; 0 1 0; 0 0 1]
    expected_transitions4 = [[1 1 1; 0 0 0; 1 1 0], [1 0 1; 0 1 0; 0 0 1]]
    fused4, transitions4 = fuse_projectors(projectors4)
    @test expected_fused4 == fused4
    @test expected_transitions4 == values(transitions4)
end
