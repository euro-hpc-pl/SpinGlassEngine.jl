foo(x, y) = x + y

for i ∈ 1:2, j ∈ 1:2
    Symbol("x$(i)$(j)") = foo($i, $j)
end
