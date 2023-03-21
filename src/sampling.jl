export 
        HLW_solution,
        decode_samples,
        EA_parameter,
        average_EA_parameter_disjoint,
        average_EA_parameter_all,
        HLW_parameter,
        HLW_correlation_on_site,
        HLW_correlation_on_site_nnn,
        average_HLW_parameter_nn,
        average_HLW_parameter_nnn,
        average_HLW_parameter_plaquette,
        measure_overlap,
        tau_r

struct HLW_solution
    HLW_nn::Matrix
    HLW_nnn::Matrix
    HLW_plaq::Real
    taur::Matrix
    EA_par_disjoint::Real 
    EA_par_all::Real
        
    function HLW_solution(HLW_nn::Matrix, HLW_nnn::Matrix, HLW_plaq::Real, taur::Matrix, EA_par_disjoint::Real, EA_par_all::Real)
        new(HLW_nn, HLW_nnn, HLW_plaq, taur, EA_par_disjoint, EA_par_all)
    end
end

function decode_samples(ctr::AbstractContractor, sol::Solution)
    fg = ctr.peps.factor_graph
    states_fg = decode_factor_graph_state.(Ref(fg), sol.states)
    states_fg = sort.(collect.(states_fg), by = x->x[1])
    s = []
    for v in states_fg
        st = [f for (i,f) in v]
        append!(s, [st])
    end
    s
end

function EA_parameter(s1::Vector{Int}, s2::Vector{Int})
    s1 = reshape(s1, :, length(s1))
    s2 = reshape(s2, :, length(s2))
    dot(s1, s2) / length(s1)
end

function HLW_parameter(s1::Vector{Int}, s2::Vector{Int})
    s1 .* s2
end

function average_EA_parameter_disjoint(s, num_states::Int)
    o_ea = 0
    count = 0
    for i in 1:2:num_states-1
        count += 1
        o_ea += EA_parameter(s[i], s[i+1])
    end
    o_ea/count
end

function average_EA_parameter_all(s, num_states::Int)
    o_ea = 0
    count = 0
    for i in 1:num_states
        for j in i + 1 : num_states
            count += 1
            o_ea += EA_parameter(s[i], s[j])
        end
    end
    o_ea/count
end

function neighbor_left(i::Int, j::Int, N::Int, r::Int)
    (i, mod(j - r - 1, N) + 1)
end

function neighbor_right(i::Int, j::Int, N::Int, r::Int)
    (i, mod(j + r - 1, N) + 1)
end

function neighbor_up(i::Int, j::Int, N::Int, r::Int)
    (mod(i - r - 1, N) + 1, j)
end

function neighbor_down(i::Int, j::Int, N::Int, r::Int)
    (mod(i + r - 1, N) + 1, j)
end

function l2k(n::Int, N::Int)
    k = CartesianIndices((N,N))[n]
    (k[2], k[1])
end

function k2l(t::Tuple, N::Int)
    (i, j) = t
    LinearIndices((N, N))[j, i]
end

#nearest neighbors
function HLW_correlation_on_site(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int, s2::Int)
    s1 = st[s1]
    s2 = st[s2]
    s = HLW_parameter(s1, s2)
    tau_c = s[i]
    (lk1, lk2) = l2k(i, n)
    tau_u = s[k2l(neighbor_up(lk1, lk2, n, 1), n)]
    tau_d = s[k2l(neighbor_down(lk1, lk2, n, 1), n)]
    tau_l = s[k2l(neighbor_left(lk1, lk2, n, 1), n)]
    tau_r = s[k2l(neighbor_right(lk1, lk2, n, 1), n)]
    (tau_c * tau_l, tau_c * tau_u, tau_c * tau_r, tau_c * tau_d)
end

#nearest neighbors
function HLW_correlation_on_site_nnn(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int, s2::Int)
    s1 = st[s1]
    s2 = st[s2]
    s = HLW_parameter(s1, s2)
    tau_c = s[i]
    (lk1, lk2) = l2k(i, n)
    tau_u = s[k2l(neighbor_up(lk1, lk2, n, 2), n)]
    tau_d = s[k2l(neighbor_down(lk1, lk2, n, 2), n)]
    tau_l = s[k2l(neighbor_left(lk1, lk2, n, 2), n)]
    tau_r = s[k2l(neighbor_right(lk1, lk2, n, 2), n)]
    (tau_c * tau_l, tau_c * tau_u, tau_c * tau_r, tau_c * tau_d)
end
    
function HLW_correlations_plaquette(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int, s2::Int)
    s1 = st[s1]
    s2 = st[s2]
    s = HLW_parameter(s1, s2)
    tau4 = s[i] * s[i+1] * s[i-n] * s[i+1-n]
    tau4
end

function tau_r(st::Vector{Vector{Int}}, i::Int, s1::Int, s2::Int, r::Int)
    s1 = st[s1]
    s2 = st[s2]
    s = HLW_parameter(s1, s2)
    tau = []
    for rr in 0:r-1
        append!(tau, s[i]*s[i+rr])
    end
    tau
end

function average_tau_r(st::Vector{Vector{Int}}, i::Int, num_states::Int, r::Int)
    t = zeros(i, r)
    R = rand(1:num_states, num_states)
    K = rand(1:num_states, num_states)
    for k in 1:i
        tr = zeros(r)
        count = 0
        for (jj, j) in enumerate(R)
            count += 1
            tr += tau_r(st, k, j, K[jj], r)
        end
        t[k,:] = tr./count
    end
    mean(t, dims=1)
end

#HLW correlations tau in space for nearest neighbors 
#(left, up, right, down from central)
function average_HLW_parameter_nn(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states), 4)
    R = rand(1:num_states, num_states)
    K = rand(1:num_states, num_states)
    for (i, k) in enumerate(R[1:end])
        tau_l, tau_u, tau_r, tau_d = [], [], [], []
        for n in 1:N*N 
            correlations = HLW_correlation_on_site(st, n, N, k, K[i])
            append!(tau_l, correlations[1])
            append!(tau_u, correlations[2])
            append!(tau_r, correlations[3])
            append!(tau_d, correlations[4])
        end
        tt[i, 1] = sum(tau_l)./length(tau_l)
        tt[i, 2] = sum(tau_u)./length(tau_u)
        tt[i, 3] = sum(tau_r)./length(tau_r)
        tt[i, 4] = sum(tau_d)./length(tau_d)
    end
    sum(tt, dims=1) ./ (num_states)
end

#HLW correlations tau in space for next nearest neighbors 
#(left, up, right, down from central)
# TODO: dispatching
function average_HLW_parameter_nnn(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states), 4)
    R = rand(1:num_states, num_states)
    K = rand(1:num_states, num_states)
    for (i, k) in enumerate(R)
        tau_l, tau_u, tau_r, tau_d = [], [], [], []
        for n in 1:N*N
            correlations = HLW_correlation_on_site_nnn(st, n, N, k, K[i])
            append!(tau_l, correlations[1])
            append!(tau_u, correlations[2])
            append!(tau_r, correlations[3])
            append!(tau_d, correlations[4])
        end
        tt[i, 1] = sum(tau_l)./length(tau_l)
        tt[i, 2] = sum(tau_u)./length(tau_u)
        tt[i, 3] = sum(tau_r)./length(tau_r)
        tt[i, 4] = sum(tau_d)./length(tau_d)
    end
    sum(tt, dims=1) ./ (num_states)
end

function average_HLW_parameter_plaquette(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states), 4)
    tau = []
    R = rand(1:num_states, num_states)
    K = rand(1:num_states, num_states)
    for j in 1:N-1
        for n in 1:N-1
            t = 0
            count = 0
            for (i, k) in enumerate(R)
                count = count + 1
                t += HLW_correlations_plaquette(st, n+j*N, N, k, K[i])
            end
            append!(tau, t/count)
        end
    end
    sum(tau)/length(tau)
end

function measure_overlap(data, N::Int, num_states)
    s = data[:,7]
    st = zeros(Int, num_states, N*N)
    
    for (j,i) in enumerate(s)
        i = replace(i, "," => "")
        i = replace(i, "[" => "", count = 1)
        i = replace(i, "]" => "", count = 1)
        for (l,ii) in enumerate(split(i))
            st[j, l] = parse(Int, ii)
        end
    end
    st = [st[i, :] for i in 1:size(st,1)]
    EA_par_disjoint = average_EA_parameter_disjoint(st, num_states)
    EA_par_all = average_EA_parameter_all(st, num_states)
    HLW_nn = average_HLW_parameter_nn(st, N, num_states)
    HLW_nnn = average_HLW_parameter_nnn(st, N, num_states)
    HLW_plaq = average_HLW_parameter_plaquette(st, N, num_states)
    taur = average_tau_r(st, Int(N/2), num_states, Int(N/2))
    HLW_solution(HLW_nn, HLW_nnn, HLW_plaq, taur, EA_par_disjoint, EA_par_all)
end