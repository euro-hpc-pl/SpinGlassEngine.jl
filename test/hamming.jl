using Test

# Define the function to test
function vector_to_integer(vector::Vector{Int})
    int_repr = sum(2^(i-1) * vector[i] for i in 1:length(vector))
    return int_repr
end

function hamming_distance(dstate::Vector{Int})
    sum(count_ones(st) for st in dstate)
end

@testset "Vector to Integer Conversion" begin
    @test vector_to_integer([0, 0, 0, 0]) == 0
    @test vector_to_integer([1, 0, 0, 0]) == 1
    @test vector_to_integer([0, 1, 0, 1]) == 10
    @test vector_to_integer([1, 1, 1, 1]) == 15
end

@testset "count_ones() method" begin
# Julia program to illustrate the use of count_ones() method
# Getting number of ones present in the binary representation of the specified number.
    for t in [0, 1, 15, 20]
        @test count_ones(t) == count(==('1'), bitstring(t))
    end
end

@testset "Hamming Distance Calculation" begin
    @test hamming_distance([10, 6, 12]) == 6
    @test hamming_distance([15, 0, 9]) == 6
    @test hamming_distance([13, 13, 13]) == 9
end

@testset "XOR Operation Tests" begin
    @test 0 ⊻ 0 == xor(0, 0) == 0
    @test 0 ⊻ 1 == xor(0, 1) == 1
    @test 1 ⊻ 0 == xor(1, 0) == 1
    @test 1 ⊻ 1 == xor(1, 1) == 0
    @test 10 ⊻ 5 == xor(10, 5) == 15 
    @test 15 ⊻ 5 == xor(15, 5) == 10
    @test 255 ⊻ 85 == xor(255, 85) == 170
    @test 170 ⊻ 85 == xor(170, 85) == 255

    println(bitstring(10))
    println(bitstring(5))
    binary_string = "1111"
    integer_value = parse(Int, binary_string, base=2)
    println(integer_value)
    @test 10 ⊻ 5 == xor(10, 5) == 15 
end