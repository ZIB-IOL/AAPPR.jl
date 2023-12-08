using SparseArrays
using Test

include("../src/page_rank_algorithms.jl")
include("../src/graphs.jl")
include("../src/objectives.jl")
include("../src/optimization_algorithms.jl")



ns = [10, 100]
edge_probabilities = [0.001, 0.2]
alphas = [0.01, 0.3]
rhos = [0.000001, 0.3]
epsilons = [0.000001, 0.1]

@testset "Test suite for accelerated_sparse_page_rank_algorithm" begin
    # Tests whether the solution is epsilon-approximate
    for n in ns, edge_probability in edge_probabilities, alpha in alphas, rho in rhos, epsilon in epsilons
        A, D = create_connected_graph(n, edge_probability)
        D_neg = D^(-1/2)
        D_pos = D^(1/2)
        Q = Symmetric(D_neg*(D - ((1-alpha)/2) * (D + A))*D_neg)
        L = 1
        mu = alpha
        s = sparsevec([1, 2, 3], [0.1, 0.2, 0.7], n)
        grad_cst_component = -alpha*D_neg*s .+ (alpha*rho)*diag(D_pos)
        grad_norm_0 =  norm(-alpha*D_neg*s) + alpha*rho*n^(3/2)
        obj = APPRObjective(Q, s, L, mu, grad_cst_component, grad_norm_0)

        function proj(x)
            return projection_onto_orthant(x)
        end
        grad_cst_indices = ∇appr_objective_cst(obj)
        x0 = spzeros(length(s))
        solution_accurate, _ = projected_gradient_descent(obj, x0, proj, 1000)
        loss_accurate = appr_objective(obj, solution_accurate)
        for use_anchor in [true, false], full_gradient_period in [nothing, 1, 10], variant in[:apgd, :cg], with_nnzs_and_losses in [true, false]
            solution_aspr, times_nnzs_and_losses = accelerated_sparse_page_rank_algorithm(obj; epsilon=epsilon, use_anchor=use_anchor, full_gradient_period=full_gradient_period, variant=variant, with_nnzs_and_losses)
            loss_aspr = appr_objective(obj, solution_aspr)
            @test loss_aspr - loss_accurate < epsilon
        end
    end
end


@testset "Test suite for the conjugate_directions_page_rank_algorithm" begin
    # Tests whether the solution is epsilon-approximate
    for n in ns, edge_probability in edge_probabilities, alpha in alphas, rho in rhos, epsilon in epsilons, exhaust_gradient in [true, false], with_nnzs_and_losses in [true, false]
        A, D = create_connected_graph(n, edge_probability)
        D_neg = D^(-1/2)
        D_pos = D^(1/2)
        Q = Symmetric(D_neg*(D - ((1-alpha)/2) * (D + A))*D_neg)
        L = 1
        mu = alpha
        s = sparsevec([1, 2, 3], [0.1, 0.2, 0.7], n)
        grad_cst_component = -alpha*D_neg*s .+ (alpha*rho)*diag(D_pos)
        grad_norm_0 =  norm(-alpha*D_neg*s) + alpha*rho*n^(3/2)
        obj = APPRObjective(Q, s, L, mu, grad_cst_component, grad_norm_0)
        solution_cdpr, times_nad_nnzs = conjugate_directions_page_rank_algorithm(obj; epsilon=epsilon, exhaust_gradient=exhaust_gradient, with_nnzs_and_losses)
        loss_cdpr = appr_objective(obj, solution_cdpr)
        function proj(x)
            return projection_onto_orthant(x)
        end
        solution_accurate, _ = projected_gradient_descent(obj, spzeros(n), proj, 1000)
        loss_accurate = appr_objective(obj, solution_accurate)
        @test loss_cdpr - loss_accurate < epsilon
    end
end


@testset "Test suite for ista" begin
    # Tests whether the solution is epsilon-approximate
    for n in ns, edge_probability in edge_probabilities, alpha in alphas, rho in rhos, epsilon in epsilons, with_nnzs_and_losses in [true, false]
        A, D = create_connected_graph(n, edge_probability)
        D_neg = D^(-1/2)
        D_pos = D^(1/2)
        Q = Symmetric(D_neg*(D - ((1-alpha)/2) * (D + A))*D_neg)
        L = 1
        mu = alpha
        s = sparsevec([1, 2, 3], [0.1, 0.2, 0.7], n)
        grad_cst_component = -alpha*D_neg*s .+ (alpha*rho)*diag(D_pos)
        grad_norm_0 =  norm(-alpha*D_neg*s) + alpha*rho*n^(3/2)
        obj = APPRObjective(Q, s, L, mu, grad_cst_component, grad_norm_0)

        function proj(x)
            return projection_onto_orthant(x)
        end
        grad_cst_indices = ∇appr_objective_cst(obj)
        x0 = spzeros(length(s))
        solution_accurate, _ = projected_gradient_descent(obj, x0, proj, 1000)
        loss_accurate = appr_objective(obj, solution_accurate)
        solution_ista, times_nnzs_and_losses = ista(obj; epsilon=epsilon, with_nnzs_and_losses)
        loss_ista = appr_objective(obj, solution_ista)
        @test loss_ista - loss_accurate < epsilon
    end
end


@testset "Test suite for fista" begin
    # Tests whether the solution is epsilon-approximate
    for n in ns, edge_probability in edge_probabilities, alpha in alphas, rho in rhos, epsilon in epsilons, with_nnzs_and_losses in [true, false]
        A, D = create_connected_graph(n, edge_probability)
        D_neg = D^(-1/2)
        D_pos = D^(1/2)
        Q = Symmetric(D_neg*(D - ((1-alpha)/2) * (D + A))*D_neg)
        L = 1
        mu = alpha
        s = sparsevec([1, 2, 3], [0.1, 0.2, 0.7], n)
        grad_cst_component = -alpha*D_neg*s .+ (alpha*rho)*diag(D_pos)

        grad_norm_0 =  norm(-alpha*D_neg*s) + alpha*rho*n^(3/2)
        obj = APPRObjective(Q, s, L, mu, grad_cst_component, grad_norm_0)

        function proj(x)
            return projection_onto_orthant(x)
        end
        grad_cst_indices = ∇appr_objective_cst(obj)
        x0 = spzeros(length(s))
        solution_accurate, _ = projected_gradient_descent(obj, x0, proj, 1000)
        loss_accurate = appr_objective(obj, solution_accurate)
        solution_fista, times_nnzs_and_losses = fista(obj; epsilon=epsilon, with_nnzs_and_losses=with_nnzs_and_losses)
        loss_fista = appr_objective(obj, solution_fista)
        @test loss_fista - loss_accurate < epsilon
    end
end
