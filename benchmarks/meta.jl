n = 2

#function foo(n::Int)
    β = 1.0
    b = rand(n)
    d = rand(n)

    for (x, y) ∈ ((:a, :(b)), (:c, :($d)))
        eval(quote
            $x = sin.($y .+ $β)
        end
        )
    end
    a, c
#end

#a, c = foo(n)

println(a)
println(c)
