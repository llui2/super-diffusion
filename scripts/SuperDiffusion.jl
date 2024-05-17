using Graphs, SimpleWeightedGraphs
using Random
using Arpack
using DataFrames
using CSV
using Statistics

using SortingAlgorithms
using LinearAlgebra
using Combinatorics
using StatsBase

# RA-PC-NC.jl

NUM_DIGITS = 8

struct Superdiffusion_Results
    model1::String
    model2::String
    N::Int64
    num_duplex::Int64
    sorted::Bool
    reversed::Bool
    ks1::Vector{Float64}
    ks2::Vector{Float64}
    df::DataFrame
end

function duplex_superdiffusion_info(G1, G2)
    L1 = laplacian_matrix(G1)
    L2 = laplacian_matrix(G2)
    La = (L1 + L2) / 2
    λ1 = eigs(L1, nev=3, which=:SM)[1][2]
    λ2 = eigs(L2, nev=3, which=:SM)[1][2]
    λa = eigs(La, nev=3, which=:SM)[1][2]
    ξ = (λa - max(λ1, λ2)) / max(λ1, λ2)
    has_superdiffusion = (ξ > 0)
    ξ = round(ξ, digits=NUM_DIGITS)
    λ1 = round(λ1, digits=NUM_DIGITS)
    λ2 = round(λ2, digits=NUM_DIGITS)
    λa = round(λa, digits=NUM_DIGITS)
    return ξ, has_superdiffusion, λa, [λ1, λ2]
end

function my_erdos_renyi(N::Int64, k::Float64, sorted::Bool, reversed::Bool)
    p = k / (N - 1)
    G = erdos_renyi(N, p)
    while !is_connected(G)
        G = erdos_renyi(N, p)
    end
    if sorted
        v_sorted = sortperm(degree(G), rev=!reversed)
        G, vmap = induced_subgraph(G, v_sorted)
    end
    return G
end

function my_random_regular(N::Int64, k::Float64,)
    k = round(Int, k)
    G = random_regular_graph(N, k)
    return G
end

function my_scale_free(N::Int64, k::Float64, sorted::Bool, reversed::Bool)
    k = round(Int, k)
    m = Int(k * N)
    G = static_scale_free(N, m, 3)
    while !is_connected(G)
        G = static_scale_free(N, m, 3)
    end
    if sorted
        v_sorted = sortperm(degree(G), rev=!reversed)
        G, vmap = induced_subgraph(G, v_sorted)
    end
    if !sorted
        rand_sort = randperm(N)
        G, vmap = induced_subgraph(G, rand_sort)
    end
    return G
end

function superdiffusion_analysis(model1::String, model2::String, N::Int64, ks1::Vector{Float64}, ks2::Vector{Float64}, num_duplex::Int64, sorted::Bool, reversed::Bool)

    if model1 ∉ ["ER", "RR", "SF"]
        throw(DomainError(model1, "Incorrect Model 1"))
    end
    if model2 ∉ ["ER", "RR", "SF"]
        throw(DomainError(model2, "Incorrect Model 2"))
    end

    rep_abort = 100
    unit_vector = ones(N)

    num_ks1 = length(ks1)
    num_ks2 = length(ks2)

    df = DataFrame(
        k1=Float64[], k2=Float64[], ka=Float64[],
        k1_avg=Float64[], k2_avg=Float64[], ka_avg=Float64[],
        ξ_avg=Float64[], prob_avg=Float64[],
        λ1_avg=Float64[], λ2_avg=Float64[], λa_avg=Float64[],
        σ_λ1=Float64[], σ_λ2=Float64[], σ_λa=Float64[],
        η_avg=Float64[], prob_η_avg=Float64[], correct=Float64[])

    for i in 1:num_ks1
        k1 = ks1[i]

        for j in 1:num_ks2
            k2 = ks2[j]

            num_duplex_shortened = num_duplex

            k1_avg, k2_avg, ka_avg = 0.0, 0.0, 0.0
            ξ_avg, prob_avg = 0.0, 0.0
            λ1_avg, λ2_avg, λa_avg = 0.0, 0.0, 0.0
            list_λ1, list_λ2, list_λa = Float64[], Float64[], Float64[]
            η_avg, prob_η_avg = 0.0, 0.0
            correct = 0.0

            for rep in 1:num_duplex

                println("\u1b[1F")
                println(string(" k1 = ", k1, " k2 = ", k2, " D = ", rep, "\u1b[1F"))

                succeeded = false
                while !succeeded

                    ### LAYER 1 ###
                    if model1 == "ER"
                        G1 = my_erdos_renyi(N, k1, sorted, false)
                    elseif model1 == "RR"
                        G1 = my_random_regular(N, k1)
                    elseif model1 == "SF"
                        G1 = my_scale_free(N, k1, sorted, false)
                    end

                    ### LAYER 2 ###
                    if model2 == "ER"
                        G2 = my_erdos_renyi(N, k2, sorted, reversed)
                    elseif model2 == "RR"
                        G2 = my_random_regular(N, k2)
                    elseif model2 == "SF"
                        G2 = my_scale_free(N, k2, sorted, reversed)
                    end

                    try
                        A1 = adjacency_matrix(G1)
                        A2 = adjacency_matrix(G2)
                        Aa = (A1 + A2) / 2
                        degs1 = A1 * unit_vector
                        degs2 = A2 * unit_vector
                        degsa = Aa * unit_vector

                        k1_avg += mean(degs1)
                        k2_avg += mean(degs2)
                        ka_avg += mean(degsa)

                        ξ, has_superdiffusion, λa, λs = duplex_superdiffusion_info(G1, G2)

                        ξ_avg += ξ
                        if has_superdiffusion
                            prob_avg += 1
                        end

                        λ1_avg += λs[1]
                        λ2_avg += λs[2]
                        λa_avg += λa

                        push!(list_λ1, λs[1])
                        push!(list_λ2, λs[2])
                        push!(list_λa, λa)

                        k1_min = minimum(degs1)
                        k2_min = minimum(degs2)
                        ka_min = minimum(degsa)

                        η = (ka_min - max(k1_min, k2_min)) / max(k1_min, k2_min)
                        η_avg += η
                        if η > 0
                            prob_η_avg += 1
                        end

                        if has_superdiffusion && η > 0
                            correct += 1
                        elseif !has_superdiffusion && η < 0
                            correct += 1
                        end

                        succeeded = true
                    catch e
                        succeeded = false
                    end
                end

                if rep == rep_abort && prob_avg == 0.0
                    num_duplex_shortened = rep_abort
                    break
                end
            end

            ka = (k1 + k2) / 2

            k1_avg /= num_duplex_shortened
            k2_avg /= num_duplex_shortened
            ka_avg /= num_duplex_shortened

            ξ_avg /= num_duplex_shortened
            prob_avg /= num_duplex_shortened

            λ1_avg /= num_duplex_shortened
            λ2_avg /= num_duplex_shortened
            λa_avg /= num_duplex_shortened

            η_avg /= num_duplex_shortened

            correct /= num_duplex_shortened

            σ_λ1 = std(list_λ1)
            σ_λ2 = std(list_λ2)
            σ_λa = std(list_λa)

            ξ_avg = round(ξ_avg, digits=NUM_DIGITS)
            prob_avg = round(prob_avg, digits=NUM_DIGITS)

            λ1_avg = round(λ1_avg, digits=NUM_DIGITS)
            λ2_avg = round(λ2_avg, digits=NUM_DIGITS)
            λa_avg = round(λa_avg, digits=NUM_DIGITS)

            σ_λ1 = round(σ_λ1, digits=NUM_DIGITS)
            σ_λ2 = round(σ_λ2, digits=NUM_DIGITS)
            σ_λa = round(σ_λa, digits=NUM_DIGITS)

            η_avg = round(η_avg, digits=NUM_DIGITS)
            prob_η_avg = round(prob_η_avg, digits=NUM_DIGITS)

            correct = round(correct, digits=NUM_DIGITS)

            push!(df, (k1, k2, ka, k1_avg, k2_avg, ka_avg, ξ_avg, prob_avg, λ1_avg, λ2_avg, λa_avg, σ_λ1, σ_λ2, σ_λa, η_avg, prob_η_avg, correct))
        end
    end

    return Superdiffusion_Results(model1, model2, N, num_duplex, sorted, reversed, ks1, ks2, df)
end

function save_results(fn::String, results::Superdiffusion_Results)
    CSV.write(fn, results.df)
end

# annealing.jl

function laplacian_eigenvalue(G1, G2)
    L1 = laplacian_matrix(G1)
    L2 = laplacian_matrix(G2)
    La = (L1 + L2) / 2
    succeeded = false
    λa = 0
    while !succeeded
        try
            λa = eigs(La, nev=3, which=:SM)[1][2]
            succeeded = true
        catch e
            succeeded  = false
        end
    end
    return λa
end

function kmin(G1, G2)
    A1 = adjacency_matrix(G1)
    A2 = adjacency_matrix(G2)
    degree1 = my_graph_degrees(A1)
    degree2 = my_graph_degrees(A2)
    degrees = (degree1 .+ degree2) / 2
    return minimum(degrees)
end

function my_graph_degrees(A)
    N = size(A, 1)
    degrees = zeros(N)
    for i in 1:N
        degrees[i] = sum(A[i, :])
    end
    return degrees
end

function generate_graphs(N, k1, k2, model1, model2)
    if model1 == "SF"
        k1 = round(Int, k1)
        m1 = Int(k1 * N / 2)
        G1 = static_scale_free(N, 2 * m1, 3)
        while !is_connected(G1)
            G1 = static_scale_free(N, 2 * m1, 3)
        end
    elseif model1 == "ER"
        G1 = erdos_renyi(N, k1 / (N - 1))
        while !is_connected(G1)
            G1 = erdos_renyi(N, k1 / (N - 1))
        end
    elseif model1 == "RR"
        G1 = random_regular_graph(N, k1)
        while !is_connected(G1)
            G1 = random_regular_graph(N, k1)
        end
    end
    if model2 == "SF"
        k2 = round(Int, k2)
        m2 = Int(k2 * N / 2)
        G2 = static_scale_free(N, 2 * m2, 3)
        while !is_connected(G2)
            G2 = static_scale_free(N, 2 * m2, 3)
        end
    elseif model2 == "ER"
        G2 = erdos_renyi(N, k2 / (N - 1))
        while !is_connected(G2)
            G2 = erdos_renyi(N, k2 / (N - 1))
        end
    elseif model2 == "RR"
        G2 = random_regular_graph(N, k2)
        while !is_connected(G2)
            G2 = random_regular_graph(N, k2)
        end
    end
    return G1, G2
end

function SA(G1, G2, MCS, T, restrict=false)
    N = nv(G1)
    σ = collect(1:N)

    G = Graph((adjacency_matrix(G1) + adjacency_matrix(G2)) / 2.0)
    δ = kmin(G1, G2)

    λ = laplacian_eigenvalue(G1, G2)

    δs = []
    λs = []

    push!(δs, δ)
    push!(λs, λ)

    for m in 1:MCS

        println("\u1b[1F")
        println("\t", round(100 * m / MCS, digits=2), " % \u1b[1F")

        δ_old = copy(δ)
        σ_new = copy(σ)
        λ_old = copy(λ)

        i = rand(1:N)
        j = rand(1:N)
        if restrict
            while degree(G2, i) != degree(G2, j)
                j = rand(1:N)
            end
        end

        σ_new[i], σ_new[j] = σ_new[j], σ_new[i]

        G2_new, _ = induced_subgraph(G2, σ_new)
        G = Graph((adjacency_matrix(G1) + adjacency_matrix(G2_new)) / 2.0)
        δ_new = kmin(G1, G2_new)
        λ_new = laplacian_eigenvalue(G1, G2_new)

        if rand() < min(1.0, exp((λ_new - λ_old) / T))
            σ = σ_new
            δ = δ_new
            λ = λ_new
            push!(δs, δ_new)
            push!(λs, λ_new)
        else
            push!(δs, δ_old)
            push!(λs, λ_old)
        end

        T = T * (1 - m / MCS)

    end
    return σ, δs, λs
end
