# Compares the performance of different algorithms w.r.t. running time.

using RegularExpressions, Serialization

include("../src/plots.jl")
include("experiment_parameters.jl")


"""
    create_time_plot(graph, algorithms, epsilon, rhos, alphas; title_prefix="all", runs=1, x_axis_scale=:log10, y_axis_scale=:log10)

Create a plot of the running times of the algorithms for the given parameters.
"""
function create_time_plot(graph, algorithms, epsilon, rhos, alphas; title_prefix="all", runs=1, x_axis_scale=:log10, y_axis_scale=:log10)
    try
        @assert length(rhos) == 1 || length(alphas) == 1 "only one of rhos and alphas can be a vector"
        if length(rhos) == 1
            x_axis = L"Regularization $\alpha$"
            figure_name = "rho_$(graph)_epsilon=$(replace(string(epsilon), "." => ","))_rho=$(replace(string(rhos[1]), "." => ","))"
            data_parameters = repeat([alphas], length(algorithms))
        else
            x_axis = L"$\rho$"
            figure_name = "alpha_$(graph)_epsilon=$(replace(string(epsilon), "." => ","))_alpha=$(replace(string(alphas[1]), "." => ","))"
            data_parameters = repeat([rhos], length(algorithms))
        end

        figure_name = string(title_prefix, "_", "time", "_", figure_name)
        y_axis = "Time in seconds"

        labels = []
        for algorithm in algorithms
            label = replace(algorithm, r"\s+" => "_")
            push!(labels, label)
        end
        labels = reshape(labels, 1, length(labels))

        algorithms = reshape(algorithms, 1, length(algorithms))
        results = open(deserialize, "results.jls")

        data_type_mean = []
        data_type_stddev = []
        println("........................................")
        println("$(graph) $(epsilon) $(alpha) $(rho) ")
        for algorithm in algorithms
            data_type = []
            for run in 1:runs, rho in rhos, alpha in alphas
                push!(data_type, results[graph][algorithm][epsilon][alpha][rho][run]["time"])
            end
            data_type = reshape(data_type, Int64(length(data_type)/runs), runs)
            push!(data_type_mean, mean(data_type, dims=2))
            push!(data_type_stddev, std(data_type, dims=2))
        end
        
        plotting_function(data_parameters, data_type_mean, labels, x_axis, y_axis; relative_path_to_figures_directory="figures/without_stddev/", figure_name=figure_name,
            save_figure=true, n_markers=nothing, x_axis_scale=x_axis_scale, y_axis_scale=y_axis_scale, legend_location=:topright)
        if runs > 1
            plotting_function(data_parameters, data_type_mean, labels, x_axis, y_axis; relative_path_to_figures_directory="figures/with_stddev/", figure_name=figure_name,
            save_figure=true, n_markers=nothing, x_axis_scale=x_axis_scale, y_axis_scale=y_axis_scale, legend_location=:topright, y_stddev=data_type_stddev)
        end
    catch e
        println("$(graph) $(epsilon) $(alpha) $(rho)")
        println("Failed to plot beacuse of error: $e")
    end
end


"""
    create_nnzs_and_losses_plot(graph, algorithms, epsilon, rho, alpha; delta=10^(-6), title_prefix="all", x_axis_scale=:log10, y_axis_scale=:identity)

Create a plot of the times and nnzs and losses at each iteration.
"""
function create_nnzs_and_losses_plot(graph, algorithms, epsilon, rho, alpha; delta=Float64(10.0^-16), title_prefix="all")
    figure_name = "$(graph)_epsilon=$(replace(string(epsilon), "." => ","))_alpha=$(replace(string(alpha), "." => ","))_rho=$(replace(string(rho), "." => ","))"

    y_axis_nnzs_iterate = "Number of nonzeros"
    y_axis_nnzs_neighborhood = "Neighborhood size"
    y_axis_losses = "Suboptimality gap"
    x_axis = "Time in seconds"

    labels = []
    for algorithm in algorithms
        label = replace(algorithm, r"\s+" => "_")
        push!(labels, label)
    end
    labels = reshape(labels, 1, length(labels))

    algorithms = reshape(algorithms, 1, length(algorithms))
    results = open(deserialize, "results.jls")
    
    x_axes = []
    y_axes_nnzs_iterate = []
    y_axes_nnzs_neighborhood = []
    y_axes_losses = []


    println("........................................")
    println("$(graph) $(epsilon) $(alpha) $(rho) ")
    minimum_loss = 10^10
    for algorithm in algorithms
        # println("results[graph][algorithm]", results[graph][algorithm])
        # println("results[graph][algorithm][epsilon]", results[graph][algorithm][epsilon])
        # println("results[graph][algorithm][epsilon][alpha]", results[graph][algorithm][epsilon][alpha])
        # println("results[graph][algorithm][epsilon][alpha][rho]", results[graph][algorithm][epsilon][alpha][rho])
        # println("results[graph][algorithm][epsilon][alpha][rho][1]", results[graph][algorithm][epsilon][alpha][rho][1])
        minimum_loss = minimum([minimum_loss, minimum(results[graph][algorithm][epsilon][alpha][rho][1]["times_nnzs_and_losses"][4][1:end])])
    end
    minimum_loss = minimum_loss - 0.01 * abs(minimum_loss)
    for algorithm in algorithms
        push!(x_axes, results[graph][algorithm][epsilon][alpha][rho][1]["times_nnzs_and_losses"][1][1:end])
        push!(y_axes_nnzs_iterate, results[graph][algorithm][epsilon][alpha][rho][1]["times_nnzs_and_losses"][2][1:end])
        push!(y_axes_nnzs_neighborhood, results[graph][algorithm][epsilon][alpha][rho][1]["times_nnzs_and_losses"][3][1:end])
        push!(y_axes_losses, results[graph][algorithm][epsilon][alpha][rho][1]["times_nnzs_and_losses"][4][1:end] .- minimum_loss)
    end
    
    try
        plotting_function(x_axes, y_axes_nnzs_iterate, labels, x_axis, y_axis_nnzs_iterate; relative_path_to_figures_directory="figures/without_stddev/", figure_name=string(title_prefix, "_", "nnzs_iterate", "_", figure_name),
            save_figure=true, n_markers=5, x_axis_scale=:log10, y_axis_scale=:identity, legend_location=:bottomright)
    catch e
        println("$(graph) $(algorithm) $(epsilon) $(alpha) $(rho)")
        println("Failed to plot nnzs_iterate beacause of error: $e")
    end
    try
        plotting_function(x_axes, y_axes_nnzs_neighborhood, labels, x_axis, y_axis_nnzs_neighborhood; relative_path_to_figures_directory="figures/without_stddev/", figure_name=string(title_prefix, "_", "nnzs_neighborhood", "_", figure_name),
            save_figure=true, n_markers=5, x_axis_scale=:log10, y_axis_scale=:identity, legend_location=:bottomright)
    catch e
        println("$(graph) $(algorithm) $(epsilon) $(alpha) $(rho)")
        println("Failed to plot nnzs_neighborhood beacause of error: $e")
    end
    try
        plotting_function(x_axes, y_axes_losses, labels, x_axis, y_axis_losses; relative_path_to_figures_directory="figures/without_stddev/", figure_name=string(title_prefix, "_", "losses", "_", figure_name),
        save_figure=true, n_markers=5, x_axis_scale=:log10, y_axis_scale=:log10, legend_location=:topright)
    catch e
        println("$(graph) $(algorithm) $(epsilon) $(alpha) $(rho)")
        println("Failed to plot losses beacause of error: $e")
    end
    
end

# # #######################################################
# # Plots
# # #######################################################

graphs = [:ca_astroph, :amazon0302, :ego_twitter_u, :patent_cit_us]
# Performance
algorithms =  ["ASPR", "CASPR", "CDPR", "FISTA", "ISTA"]
for graph in graphs
    # create_time_plot(graph, algorithms, epsilon, rhos, [alpha], title_prefix="performance", runs=1)
    create_time_plot(graph, algorithms, epsilon, [rho], alphas, title_prefix="performance", runs=1)
end

# nnzs and losses
algorithms =  ["ASPR", "CASPR", "CDPR", "FISTA", "ISTA"]
for graph in graphs, alpha in alphas_nnzs_and_losses, rho in rhos_nnzs_and_losses
    create_nnzs_and_losses_plot(graph, algorithms, epsilon, rho, alpha, title_prefix="vs_time")
end


graphs = [:amazon0302, :ca_astroph]
# ASPR anchor
algorithms =  ["ASPR", "ASPR use anchor"]
for graph in graphs
    create_time_plot(graph, algorithms, epsilon, rhos, [alpha], title_prefix="ASPR_anchor", runs=1)
    create_time_plot(graph, algorithms, epsilon, [rho], alphas, title_prefix="ASPR_anchor", runs=1)
end

# ASPR full gradient
algorithms =  ["ASPR", "ASPR 1", "ASPR 5", "ASPR 15"]
for graph in graphs
    create_time_plot(graph, algorithms, epsilon, rhos, [alpha], title_prefix="ASPR_full_grad", runs=1)
    create_time_plot(graph, algorithms, epsilon, [rho], alphas, title_prefix="ASPR_full_grad", runs=1)
end

# CASPR full gradient
algorithms =  ["CASPR", "CASPR 1", "CASPR 5", "CASPR 15"]
for graph in graphs
    create_time_plot(graph, algorithms, epsilon, rhos, [alpha], title_prefix="CASPR_full_grad", runs=1)
    create_time_plot(graph, algorithms, epsilon, [rho], alphas, title_prefix="CASPR_full_grad", runs=1)
end

# CDPR recompute gradient
algorithms =  ["CDPR", "CDPR recompute gradient"]
for graph in graphs
    create_time_plot(graph, algorithms, epsilon, rhos, [alpha], title_prefix="CDPR_recompute_grad", runs=1)
    create_time_plot(graph, algorithms, epsilon, [rho], alphas, title_prefix="CDPR_recompute_grad", runs=1)
end

