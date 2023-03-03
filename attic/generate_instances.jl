using Random
using Distributions
using DelimitedFiles
using Printf

# function generate_instances(m::Int, n::Int)
#     d = Normal()
#     for i in 0:n-1
#         for j in 1:m-1
#             println(j+i*n, " ", (j+i*n)+1, " ", rand(d, 1))
#         end
#     end

#     for i in 1:m*n-n
#         println(i, " ", i + n, " ", rand(d, 1))
#     end
# end

function generate_instances(m::Int, n::Int, i::Int)
    d = Normal()
    open("test/instances/square_gauss/S$(m)/$(i).txt", "w") do file
        for i in 0:n-1
            for j in 1:m-1
                a = j+i*n
                b = (j+i*n)+1
                c = rand(d, 1)
                println(file, a, " ", b, " ", c[begin])
            end
        end

        for i in 1:m*n-n
            a = i
            b = i + n
            c = rand(d, 1)
            println(file, a, " ", b, " ", c[begin])
        end
    end
end

for i in 100:100
    # generate_instances(4, 4, i)
    # generate_instances(6, 6, i)
    generate_instances(8, 8, i)
    # generate_instances(10, 10, i)
    # generate_instances(12, 12, i)
    # generate_instances(16, 16, i)
end