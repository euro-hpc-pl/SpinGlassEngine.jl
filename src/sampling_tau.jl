export 
        HLW_correlation_on_site_tau,
        HLW_correlation_on_site_nnn_tau,
        average_HLW_parameter_nn_tau,
        average_HLW_parameter_nnn_tau,
        average_HLW_parameter_plaquette_tau,
        measure_overlap_tau,
        tau_r_tau

#nearest neighbors
function HLW_correlation_on_site_tau(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int)
    s = st[s1]
    tau_c = s[i]
    (lk1, lk2) = l2k(i, n)
    tau_u = s[k2l(neighbor_up(lk1, lk2, n, 1), n)]
    tau_d = s[k2l(neighbor_down(lk1, lk2, n, 1), n)]
    tau_l = s[k2l(neighbor_left(lk1, lk2, n, 1), n)]
    tau_r = s[k2l(neighbor_right(lk1, lk2, n, 1), n)]
    (tau_c * tau_l, tau_c * tau_u, tau_c * tau_r, tau_c * tau_d)
end

#nearest neighbors
function HLW_correlation_on_site_nnn_tau(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int)
    s = st[s1]
    tau_c = s[i]
    (lk1, lk2) = l2k(i, n)
    tau_u = s[k2l(neighbor_up(lk1, lk2, n, 2), n)]
    tau_d = s[k2l(neighbor_down(lk1, lk2, n, 2), n)]
    tau_l = s[k2l(neighbor_left(lk1, lk2, n, 2), n)]
    tau_r = s[k2l(neighbor_right(lk1, lk2, n, 2), n)]
    (tau_c * tau_l, tau_c * tau_u, tau_c * tau_r, tau_c * tau_d)
end
    
function HLW_correlations_plaquette_tau(st::Vector{Vector{Int}}, i::Int, n::Int, s1::Int)
    s = st[s1]
    tau4 = s[i] * s[i+1] * s[i-n] * s[i+1-n]
    tau4
end

function tau_r_tau(st::Vector{Vector{Int}}, i::Int, s1::Int,  r::Int)
    s = st[s1]
    tau = []
    for rr in 0:r-1
        append!(tau, s[i]*s[i+rr])
    end
    tau
end

function average_tau_r_tau(st::Vector{Vector{Int}}, i::Int, num_states::Int, r::Int)
    t = zeros(i, r)
    R = rand(1:num_states, num_states)
    for k in 1:i
        tr = zeros(r)
        count = 0
        for (jj, j) in enumerate(R)
            count += 1
            tr += tau_r(st, k, j, r)
        end
        t[k,:] = tr./count
    end
    mean(t, dims=1)
end

#HLW correlations tau in space for nearest neighbors 
#(left, up, right, down from central)
function average_HLW_parameter_nn_tau(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states), 4)
    R = rand(1:num_states, num_states)
    for (i, k) in enumerate(R[1:end])
        tau_l, tau_u, tau_r, tau_d = [], [], [], []
        for n in 1:N*N 
            correlations = HLW_correlation_on_site_tau(st, n, N, k)
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
function average_HLW_parameter_nnn_tau(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states), 4)
    R = rand(1:num_states, num_states)
    for (i, k) in enumerate(R)
        tau_l, tau_u, tau_r, tau_d = [], [], [], []
        for n in 1:N*N
            correlations = HLW_correlation_on_site_nnn_tau(st, n, N, k)
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

function average_HLW_parameter_plaquette_tau(st, N::Int, num_states::Int)
    tt = zeros(Int(num_states), 4)
    tau = []
    R = rand(1:num_states, num_states)
    for j in 1:N-1
        for n in 1:N-1
            t = 0
            count = 0
            for (i, k) in enumerate(R)
                count = count + 1
                t += HLW_correlations_plaquette_tau(st, n+j*N, N, k)
            end
            append!(tau, t/count)
        end
    end
    sum(tau)/length(tau)
end

function measure_overlap_tau(data, N::Int, num_states)
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
    HLW_nn = average_HLW_parameter_nn_tau(st, N, num_states)
    HLW_nnn = average_HLW_parameter_nnn_tau(st, N, num_states)
    HLW_plaq = average_HLW_parameter_plaquette_tau(st, N, num_states)
    taur = average_tau_r_tau(st, Int(N/2), num_states, Int(N/2))
    HLW_solution(HLW_nn, HLW_nnn, HLW_plaq, taur, EA_par_disjoint, EA_par_all)
end