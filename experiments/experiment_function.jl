# Compares the performance of different algorithms w.r.t. running time.

using LinearAlgebra, SparseArrays, Graphs, RegularExpressions, Serialization

include("../src/datasets.jl")
include("../src/optimization_algorithms.jl")
include("../src/projections.jl")
include("../src/objectives.jl")
include("../src/page_rank_algorithms.jl")
include("../src/m_matrices.jl")


function compute_runs(graph, algorithms, epsilons, rhos, alphas; runs=1, with_nnzs_and_losses=false)
    algorithms = reshape(algorithms, 1, length(algorithms))
    (g, A, D) = loadsnap64(graph)
    println("graph: $(graph)")
    println("number of vertices: $(nv(g))")
    println("number of edges: $(ne(g))")
    _, min_ind = findmin(D)

    n = nv(g)
    s = spzeros(n)
    s[min_ind[1]] = 1.
    
    L = 1
    D_neg = D^(-1/2)
    D_pos = D^(1/2)

    if isfile("results.jls")
        results = open(deserialize, "results.jls")
    else
        results = Dict()
    end

    for run in 1:runs, algorithm in algorithms, rho in rhos, alpha in alphas, epsilon in epsilons
        println("run: $(run), algorithm: $(algorithm), rho: $(rho), alpha: $(alpha), epsilon: $(epsilon)")
        Q = Symmetric(D_neg*(D - ((1-alpha)/2) * (D + A))*D_neg)
        mu = alpha
        grad_cst_component = -alpha*D_neg*s .+ (alpha*rho)*diag(D_pos)
        grad_norm_0 =  norm(-alpha*D_neg*s) + alpha*rho*n^(3/2)
        obj = APPRObjective(Q, s, L, mu, grad_cst_component, grad_norm_0)

        if occursin("CASPR", algorithm)
            if match(r"\d+", algorithm) === nothing
                full_gradient_period = nothing
            else
                full_gradient_period = parse(Int, match(r"\d+", algorithm).match)
            end
            # compile the function to avoid compilation time in the timing
            x, times_nnzs_and_losses = accelerated_sparse_page_rank_algorithm(obj; epsilon=0.9, full_gradient_period=full_gradient_period, variant=:cg, with_nnzs_and_losses=with_nnzs_and_losses)
            time_start = time()
            x, times_nnzs_and_losses = accelerated_sparse_page_rank_algorithm(obj; epsilon=epsilon, full_gradient_period=full_gradient_period, variant=:cg, with_nnzs_and_losses=with_nnzs_and_losses)
            time_elapsed = time() - time_start
        end

        if occursin("CDPR", algorithm)
            if occursin("recompute gradient", algorithm)
                exhaust_gradient = false
            else
                exhaust_gradient = true
            end
            # compile the function to avoid compilation time in the timing
            x, times_nnzs_and_losses = conjugate_directions_page_rank_algorithm(obj; epsilon=10.0^16, exhaust_gradient=exhaust_gradient, with_nnzs_and_losses=with_nnzs_and_losses)
            time_start = time()
            x, times_nnzs_and_losses = conjugate_directions_page_rank_algorithm(obj; epsilon=epsilon, exhaust_gradient=exhaust_gradient, with_nnzs_and_losses=with_nnzs_and_losses)
            time_elapsed = time() - time_start
        end
         
        if algorithm == "ISTA"
            # compile the function to avoid compilation time in the timing
            x, times_nnzs_and_losses = ista(obj; epsilon=0.9, with_nnzs_and_losses=with_nnzs_and_losses)
            time_start = time()
            x, times_nnzs_and_losses = ista(obj; epsilon=epsilon, with_nnzs_and_losses=with_nnzs_and_losses)
            time_elapsed = time() - time_start
        end

        if algorithm == "FISTA"
            # compile the function to avoid compilation time in the timing
            x, times_nnzs_and_losses = fista(obj; epsilon=0.9, with_nnzs_and_losses=with_nnzs_and_losses)
            time_start = time()
            x, times_nnzs_and_losses = fista(obj; epsilon=epsilon, with_nnzs_and_losses=with_nnzs_and_losses)
            time_elapsed = time() - time_start
        end

        if occursin("ASPR", algorithm) && !occursin("CASPR", algorithm)
            if occursin("use anchor", algorithm)
                use_anchor = true
            else
                use_anchor = false
            end
            if match(r"\d+", algorithm) === nothing
                full_gradient_period = nothing
            else
                full_gradient_period = parse(Int, match(r"\d+", algorithm).match)
            end
            # compile the function to avoid compilation time in the timing
            x, times_nnzs_and_losses = accelerated_sparse_page_rank_algorithm(obj; epsilon=0.9, use_anchor=use_anchor, full_gradient_period=full_gradient_period, variant=:apgd, with_nnzs_and_losses=with_nnzs_and_losses)
            time_start = time()
            x, times_nnzs_and_losses = accelerated_sparse_page_rank_algorithm(obj; epsilon=epsilon, use_anchor=use_anchor, full_gradient_period=full_gradient_period, variant=:apgd, with_nnzs_and_losses=with_nnzs_and_losses)
            time_elapsed = time() - time_start
        end

        if !haskey(results, graph)
            results[graph] = Dict()
        end
        if !haskey(results[graph], algorithm)
            results[graph][algorithm] = Dict()
        end
        if !haskey(results[graph][algorithm], epsilon)
            results[graph][algorithm][epsilon] = Dict()
        end
        if !haskey(results[graph][algorithm][epsilon], alpha)
            results[graph][algorithm][epsilon][alpha] = Dict()
        end
        if !haskey(results[graph][algorithm][epsilon][alpha], rho)
            results[graph][algorithm][epsilon][alpha][rho] = Dict()
        end
        if !haskey(results[graph][algorithm][epsilon][alpha][rho], run)
            results[graph][algorithm][epsilon][alpha][rho][run] = Dict()
        end

        try
            if with_nnzs_and_losses
                results[graph][algorithm][epsilon][alpha][rho][run]["times_nnzs_and_losses"] = times_nnzs_and_losses
            else
                results[graph][algorithm][epsilon][alpha][rho][run]["time"] = time_elapsed
            end
            open("results.jls", "w") do f
                serialize(f, results)
            end
        catch
            println("Could not append the results to the file.")
        end
    end
end



