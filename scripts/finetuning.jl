include("scripts/SuperDiffusion.jl")

# ======== Initializations ================================

seed = rand(1:999)
seed = 668 # seed for reproducibility
Random.seed!(seed)
println("seed = ", seed)

N = 500
k1 = 25
k2 = 50
MCS = 1e4
# Temps = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0] # Free SA ¡¡problematic!! # the lower the better because preserving degree is better, but then the explored spaces is to narrow to compare performance.
Temps = [1e-2,1e-1,1,10,20] # Restricted SA # "high" (>= 1) temperatures perform similarly
D = length(Temps)
println("#D = ", D)

G1, G2 = generate_graphs(N, k1, k2, "ER", "ER")

sort1 = sortperm(degree(G1), rev=false)
G1 = induced_subgraph(G1, sort1)[1]

sort2 = sortperm(degree(G2), rev=true)
G2_0 = induced_subgraph(G2, sort2)[1]

# ======== Simulated Annealing ================================

perms = []
lambdas = []
sigmas = []
final_lambdas = []

for i in 1:D
    local Temp = Temps[i]

    println("\u1b[1F")
    println(string("D = ", i, "\u1b[1F"))

    # σ, δs, λs = SA(G1, G2_0, MCS, Temp, false) # Free SA
    σ, δs, λs = SA(G1, G2_0, MCS, Temp, true) # Restricted SA

    push!(perms, σ)
    push!(lambdas, λs)
    push!(sigmas, δs)
    push!(final_lambdas, λs[end])
end

i_best = argmax(final_lambdas)
T_best = Temps[i_best]
info = DataFrame(i_best=i_best, T_best=T_best)
CSV.write("tuning-info.csv", info)
best = perms[i_best]

df = DataFrame(t=0:MCS)
for i in 1:D
    df[!, "λ$i"] = lambdas[i]
    df[!, "δ$i"] = sigmas[i]
end
CSV.write("data/tuning.csv", df)

# run(`python3 scripts/finetunning-plot.py`)