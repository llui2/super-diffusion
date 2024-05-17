include("SuperDiffusion.jl")

# ======== Initializations ================================

cd(Base.dirname(Base.source_path()))
pwd()

Random.seed!(555)

N = 500
num_duplex = 50

ks = collect(5.0:1.0:120.0)

# ======== Main ===========================================

# -------- Parameters --------

model1 = "ER"
model2 = "ER"

sorted = false
reversed = false

fn_results_er = string(model1, "-", model2, "-N", N, "-D", num_duplex, "-S", sorted ? "1" : "0", "-R", reversed ? "1" : "0", ".csv")
println(fn_results_er)

# -------- Run --------

@time fn_results_er (
    results_er = superdiffusion_analysis(model1, model2, N, ks, ks, num_duplex, sorted, reversed)
)
save_results(fn_results_er, results_er)

# -------- Plot --------

run(`python fig2-RA-PC-NC-plot.py`)
