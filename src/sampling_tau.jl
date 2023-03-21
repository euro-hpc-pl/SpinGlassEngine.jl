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

#nearest neighbors
function HLW_correlation_on_site(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int)
    s = st[s1]
    tau_c = s[i]
    if i in collect(1:n)
        tau_u = s[i+n*(n-1)]
        i == 1 ? tau_l = s[n] : tau_l = s[i-1]
        i == n ? tau_r = s[1] : tau_r = s[i+1]
        tau_d = s[i+n]
    elseif i in collect(n*n-n+1 : n*n)
        tau_d = s[i-n*(n-1)]
        i == n*n-n+1 ? tau_l = s[n] : tau_l = s[i-1]
        i == n*n ? tau_r = s[n*n-n+1] : tau_r = s[i+1]
        tau_u = s[i-n]
    elseif i in collect(n+1:n:n*n-2*n+1)
        tau_l = s[i+n-1]
        tau_r = s[i+1]
        tau_u, tau_d = s[i-n], s[i+n]
    elseif i in collect(2*n:n:n*n-n)
        tau_r = s[i-n+1]
        tau_l = s[i-1]
        tau_u, tau_d = s[i-n], s[i+n]
    else
        tau_l, tau_r = s[i-1], s[i+1]
        tau_u, tau_d = s[i-n], s[i+n]
    end
    (tau_c * tau_l, tau_c * tau_u, tau_c * tau_r, tau_c * tau_d)
end

#next nearest neighbors
function HLW_correlation_on_site_nnn(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int)
    s = st[s1]
    tau_c = s[i]
    if i in collect(1:2*n)
        tau_u = s[n*n + (i-2*n)]
        i in [1, 2, n+1, n+2] ? tau_l = s[i+n-2] : tau_l = s[i-2]
        i in [n, n-1, 2*n, 2*n-1] ? tau_r = s[i-n+2] : tau_r = s[i+2]
        tau_d = s[i+2*n]
    elseif i in collect(n*n-2*n+1 : n*n)
        tau_d = s[i-n*(n-1)+n]
        i in [n*n-2*n+1, n*n-2*n+2, n*n-n+1, n*n-n+2] ? tau_l = s[i+n-2] : tau_l = s[i-2]
        i in [n*n, n*n-1, n*n-n, n*n-n-1] ? tau_r = s[i-n+2] : tau_r = s[i+2]
        tau_u = s[i-n]
    elseif i in collect(n+1:n:n*n-2*n+1) || i in collect(n+2:n:n*n-2*n+2)
        tau_l = s[i+n-2]
        tau_r = s[i+2]
        tau_u, tau_d = s[i-2*n], s[i+2*n]
    elseif i in collect(2*n:n:n*n-n) || i in collect(2*n-1:n:n*n-n-1)
        tau_r = s[i-n+2]
        tau_l = s[i-2]
        tau_u, tau_d = s[i-2*n], s[i+2*n]
    else
        tau_l, tau_r = s[i-2], s[i+2]
        tau_u, tau_d = s[i-2*n], s[i+2*n]
    end
    (tau_c * tau_l, tau_c * tau_u, tau_c * tau_r, tau_c * tau_d)
end

function HLW_correlations_plaquette(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int)
    s = st[s1]
    tau4 = s[i] * s[i+1] * s[i-n] * s[i+1-n]
    tau4
end

function tau_r(st::Vector{Vector{Int}}, i::Int, s1::Int, r::Int)
    s = st[s1]

    tau = []
    for rr in 0:r-1
        append!(tau, s[i]*s[i+rr])
    end
    tau
end

function average_tau_r(st::Vector{Vector{Int}}, i::Int, num_states::Int, r::Int)
    t = zeros(i, r)
    for k in 1:i
        tr = zeros(r)
        count = 0
        for j in 1:2:num_states
            count += 1
            tr += tau_r(st, k, j, r)
        end
        t[k,:] = tr./count
    end
    mean(t, dims=1)
end

#HLW correlations tau in space for nearest neighbors 
#(left, up, right, down from central)
function average_HLW_parameter_nn(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states/2), 4)
    
    for (i, k) in enumerate(1:2:num_states)
        tau_l, tau_u, tau_r, tau_d = [], [], [], []
        for n in 1:N*N 
            correlations = HLW_correlation_on_site(st, n, N, k)
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
    sum(tt, dims=1) ./ (num_states/2)
end

#HLW correlations tau in space for next nearest neighbors 
#(left, up, right, down from central)
# TODO: dispatching
function average_HLW_parameter_nnn(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states/2), 4)
    for (i, k) in enumerate(1:2:num_states)
        tau_l, tau_u, tau_r, tau_d = [], [], [], []
        for n in 1:N*N
            correlations = HLW_correlation_on_site_nnn(st, n, N, k)
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
    sum(tt, dims=1) ./ (num_states/2)
end

function average_HLW_parameter_plaquette(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states/2), 4)
    tau = []
    for j in 1:N-1
        for n in 1:N-1
            t = 0
            count = 0
            for (i, k) in enumerate(1:2:num_states)
                count = count + 1
                t += HLW_correlations_plaquette(st, n+j*N, N, k)
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