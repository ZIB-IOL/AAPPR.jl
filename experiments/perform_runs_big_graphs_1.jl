include("experiment_function.jl")
include("experiment_parameters.jl")


# #######################################################
# Experiments
# #######################################################

# Performance
algorithms =  ["ASPR", "CASPR", "CDPR", "FISTA", "ISTA"]
graphs = [:patent_cit_us]
for graph in graphs
    # compute_runs(graph, algorithms, [epsilon], rhos, [alpha], runs=1)
    compute_runs(graph, algorithms, [epsilon], [rho], alphas, runs=1)
end

# # nnzs and losses
# algorithms =  ["ASPR", "CASPR", "CDPR", "FISTA", "ISTA"]
# graphs = [:patent_cit_us]
# for graph in graphs, alpha in alphas_nnzs_and_losses, rho in rhos_nnzs_and_losses 
#     compute_runs(graph, algorithms, [epsilon], rho, alpha, runs=1, with_nnzs_and_losses=true)
# end




