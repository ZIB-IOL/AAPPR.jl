using Test
using Arpack, LinearAlgebra, SparseArrays

include("../src/graphs.jl")
include("../src/optimization_algorithms.jl")
include("../src/projections.jl")
include("../src/objectives.jl")
include("../src/m_matrices.jl")



"""
    test_optimization_algorithms(n::Int64, percentage_indices::Float64)

Tests whether the solution of the optimization problem is epsilon-approximate for the following algorithms:
- projected_gradient_descent
- accelerated_projected_gradient_descent
- conjugate_gradients
"""
function test_optimization_algorithms(n::Int64, percentage_indices::Float64)
    if n < 5
        throw(ArgumentError("n ≥ 5 is required."))
    end
    if percentage_indices < 0 || percentage_indices > 1
        throw(ArgumentError("percentage_indices in [0, 1] is required."))
    end



    grad_norm_tol = 0.000001

    edge_probability = 0.1
    alpha = 0.1
    rho = 0.0001
    A, D = create_connected_graph(n, edge_probability)
    D_neg = D^(-1/2)

    D_pos = D^(1/2)

    Q = D_neg*(D - ((1-alpha)/2) * (D + A))*D_neg
    @assert is_m_matrix(Q)
    L = maximum(eigs(Q, nev=1, which=:LM)[1])
    mu = minimum(eigs(Q, nev=1, which=:SM)[1])
    Q = Symmetric(Q)
    x0 = spzeros(n)

    s = sparsevec([1, 2, 3, 4, 5], [0.2, 0.2, 0.2, 0.2, 0.2], n)
    grad_cst_component = -alpha*D_neg*s .+ (alpha*rho)*diag(D_pos)
    grad_norm_0 =  norm(-alpha*D_neg*s) + alpha*rho*n^(3/2)
    obj = APPRObjective(Q, s, L, mu, grad_cst_component, grad_norm_0)

    indices = collect(1:n)
    mask = rand(length(indices)) .> percentage_indices
    indices = indices[mask]
    indices = sort(unique(vcat(indices, [1, 2, 3, 4, 5])))
    objective_indices = copy(obj)
    restrict!(objective_indices, collect(1:n), indices)

    function proj(x)
        function identity_function(x)
            return x
        end
        return identity_function(x)
    end
    solution_pgd, _ = projected_gradient_descent(objective_indices, x0, proj, 1000)
    solution_apgd, _, _ = accelerated_projected_gradient_descent(objective_indices, x0, proj, 1000000; grad_norm_tol=grad_norm_tol)
    solution_cg, _, _ = conjugate_gradients(objective_indices, x0, length(x0), grad_norm_tol=grad_norm_tol)
    @test norm(solution_pgd.-solution_apgd) ≤ 10^(-5)
    @test norm(solution_cg.-solution_apgd) ≤ 10^(-5)
end


@testset "Test suite for accelerated_projected_gradient_descent, projected_gradient_descent, and conjugate_gradients" begin
    for n in [10, 250], percentage_indices in [0.01, 0.2]
        @testset "n = $n, percentage_indices = $percentage_indices" begin
            test_optimization_algorithms(n, percentage_indices)
        end
    end

end




