
include("objectives.jl")
include("projections.jl")

"""
    projected_gradient_descent(x0::SparseVector{Float64, Int64}, objective::APPRObjective, grad_cst_indices::SparseVector{Float64, Int64}, projection_oracle::Function, T::Int64)

Performs projected gradient descent on a given objective function over a feasible region defined by a set of indices.
Let C = anchor + (span {e_i : i in `indices`} ∩ R^n_{≥0}) be the feasible region.

# Arguments
- `objective::APPRObjective`: An object representing the objective function to be minimized. Must be of type `APPRObjective`.
- `x0::SparseVector{Float64, Int64}`: The starting point.
- `projection_oracle::Function`: A function that projects a given vector onto the feasible region `C`.
- `T::Int64`: The number of iterations to perform.

# Returns
- `x::SparseVector{Float64, Int64}`: The solution vector minimizing the objective function.
- `grad::SparseVector{Float64, Int64}`: The gradient of the objective function at `x`.

# References
- [1] Yurii Nesterov. “Introductory lectures on convex programming volume I: Basic course”. In: Lecture notes 3.4 (1998).
"""
function projected_gradient_descent(objective::APPRObjective, x0::SparseVector{Float64, Int64}, projection_oracle::Function, T::Int64)
    x = x0
    for t = 1:T
        x = projection_oracle(x .- 1/objective.L*(∇appr_objective(objective, x)))
    end
    return x, ∇appr_objective(objective, x)
end


"""
    accelerated_projected_gradient_descent(
        objective_indices::APPRObjective,
        x0::SparseVector{Float64, Int64},
        projection_oracle::Function,
        T::Int64;
        full_gradient_period::Union{Int64, Nothing}=nothing,
        objective_neighborhood_indices::Union{APPRObjective, Nothing}=nothing,
        grad_norm_tol::Union{Float64, Nothing}=nothing,
        indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing,
        neighborhood_indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing
        )

Performs accelerated projected gradient descent on a given objective function over a feasible region defined by a set of indices `indices`.
Let `C` be the feasible region defined as `anchor + (span {eᵢ : i in `indices`} ∩ R^n_{≥0})`.

# Arguments
- `objective_indices::APPRObjective`: An object representing the objective function to be minimized. Must be of type `APPRObjective`.
- `x0::SparseVector{Float64, Int64}`: The starting point.
- `projection_oracle::Function`: A function that projects a given vector onto the feasible region `C`.
- `T::Int64`: The number of iterations to perform.
- `grad_norm_tol::Union{Float64, Nothing}=nothing`: If provided, stop the algorithm if the norm of the gradient of the objective function at the current iterate is less than `grad_norm_tol`.
- `full_gradient_period::Union{Int64, Nothing}=nothing`: If provided, compute the full gradient of the objective function every `full_gradient_period` iterations.
- `objective_neighborhood_indices::Union{APPRObjective, Nothing}=nothing`: If provided with `full_gradient_period`, an `APPRObjective` object that provides the necessary objective function and its gradient restricted to the feasible region `C` corresponding to the neighborhood of `indices`.
- `indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing`: The set of indices `indices` defining the feasible region `C`.
- `neighborhood_indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing`: The set of indices corresponding to the neighborhood of `indices`.

# Returns
- `y::SparseVector{Float64, Int64}`: The solution vector minimizing the objective function.
- `grad_neighborhood_indices::SparseVector{Float64, Int64}`: The gradient of the objective function at `y`. (Only correct if `neighborhood_indices` were passed as a function argument.)
- `flag::Bool`: If `full_gradient_period`, `true` indicates that we do not have to reduce_and_clip the gradient in the containing algorithm.

# References
- [1] Yurii Nesterov. “Introductory lectures on convex programming volume I: Basic course”. In: Lecture notes 3.4 (1998).
"""
function accelerated_projected_gradient_descent(
    objective_indices::APPRObjective,
    x0::SparseVector{Float64, Int64},
    projection_oracle::Function,
    T::Int64;
    full_gradient_period::Union{Int64, Nothing}=nothing,
    objective_neighborhood_indices::Union{APPRObjective, Nothing}=nothing,
    grad_norm_tol::Union{Float64, Nothing}=nothing,
    indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing,
    neighborhood_indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing
    )
    set_diff_indices = Int64[]
    if full_gradient_period !== nothing
        @assert objective_neighborhood_indices !== nothing
        @assert indices !== nothing
        @assert neighborhood_indices !== nothing
        set_diff_indices = setdiff(neighborhood_indices, indices)
        objective_set_diff_indices = copy(objective_neighborhood_indices)
        set_nonindex_entries_to_zero!(objective_set_diff_indices.Q, set_diff_indices)
        set_nonindex_entries_to_zero!(objective_set_diff_indices.s, set_diff_indices)
        set_nonindex_entries_to_zero!(objective_set_diff_indices.grad_cst_component, set_diff_indices)
    end

    flag = false
    L = objective_indices.L
    mu = objective_indices.mu
    kappa = L/mu
    
    A_old = 0
    a = 1
    z = projection_oracle(x0)
    y = copy(z)
    grad_indices = spzeros(length(y))
    grad_neighborhood_indices = copy(grad_indices)

    for t in 1:T
        A_new = A_old + a
        x = (A_old/A_new)*y .+ (a/A_new)*z
        z = projection_oracle((kappa-1+A_old)/(kappa-1+A_new)*z .+ (a/(kappa-1+A_new))*(x.-(1/mu)*∇appr_objective(objective_indices, x)))
        y = (A_old/A_new)*y .+ (a/A_new)*z
        a = A_new*(((2*kappa)/(2*kappa+1-sqrt(1+4*kappa)))-1)
        A_old = A_new

        # stop if the norm of the gradient of the objective function is less than `grad_norm_tol`
        grad_indices = ∇appr_objective(objective_indices, y)
        if grad_norm_tol !== nothing && norm(grad_indices, 2) < grad_norm_tol
            break
        end

        if length(set_diff_indices) != 0 && full_gradient_period !== nothing && t % full_gradient_period == 0 && all(grad_indices[y .> 0] .< 0)
            grad_neighborhood_indices = copy(grad_indices)
            grad_neighborhood_indices[set_diff_indices] = ∇appr_objective(objective_set_diff_indices, y)[set_diff_indices]
            if length(unique(vcat(findall(x -> x < 0, grad_neighborhood_indices), indices))) > length(indices)
                flag = true
                break
            end
        end
    end

    return y, grad_neighborhood_indices, flag
end

"""
    conjugate_gradients(objective_indices::APPRObjective,
    x0::SparseVector{Float64, Int64},
    T::Int64;
    grad_norm_tol::Union{Float64, Nothing}=nothing,
    full_gradient_period::Union{Int64, Nothing}=nothing,
    objective_neighborhood_indices::Union{APPRObjective, Nothing}=nothing,
    indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing,
    neighborhood_indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing
    )

Conjugate gradient method (CG) for minimizing a quadratic objective function. If `full_gradient_period` is a positive Int64, checks wheter the current iterate projected onto the feasible region
C = anchor + (span {eᵢ : i in `indices`} ∩ R^n_{≥0}) already allows us to determine new good coordinates for PageRank.

# Arguments
- `objective_indices::APPRObjective`: An object representing the objective function to be minimized. Must be of type `APPRObjective`.
- `x0::SparseVector{Float64, Int64}`: The starting point.
- `T::Int64`: The number of iterations to perform.
- `full_gradient_period::Union{Int64, Nothing}=nothing`: If provided, compute the full gradient of the objective function every `full_gradient_period` iterations.
- `objective_neighborhood_indices::Union{APPRObjective, Nothing}=nothing`: If provided with `full_gradient_period`, an `APPRObjective` object that provides the necessary objective function and its gradient restricted to the feasible region `C` corresponding to the neighborhood of `indices`.
- `grad_norm_tol::Union{Float64, Nothing}=nothing`: If provided, stop the algorithm if the norm of the gradient of the objective function at the current iterate is less than `grad_norm_tol`.
- `indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing`: The set of indices `indices` defining the feasible region `C`.
- `neighborhood_indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing`: The set of indices corresponding to the neighborhood of `indices`. (Only correct if `neighborhood_indices` were passed as a function argument.)

# Returns
- `x::SparseVector{Float64, Int64}`: The solution vector minimizing the objective function.
- `grad_neighborhood_indices::SparseVector{Float64, Int64}`: The gradient of the objective function at `x`.
- `flag::Bool`: If `full_gradient_period`, `true` indicates that we do not have to reduce_and_clip the gradient in the containing algorithm.

# References
- [1] Sahar Karimi and Stephen A Vavasis. “A unified convergence bound for conjugate gradient and accelerated gradient”. In: arXiv preprint arXiv:1605.00320 (2016).
"""
function conjugate_gradients(
    objective_indices::APPRObjective,
    x0::SparseVector{Float64, Int64},
    T::Int64;
    grad_norm_tol::Union{Float64, Nothing}=nothing,
    full_gradient_period::Union{Int64, Nothing}=nothing,
    objective_neighborhood_indices::Union{APPRObjective, Nothing}=nothing,
    indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing,
    neighborhood_indices::Union{Nothing, AbstractVector{Any}, AbstractVector{Int64}}=nothing
    )
    @assert T ≤ length(x0) "T must be less than or equal to the length of x0"

    set_diff_indices = Int64[]
    if full_gradient_period !== nothing
        @assert objective_neighborhood_indices !== nothing
        @assert indices !== nothing
        @assert neighborhood_indices !== nothing
        set_diff_indices = setdiff(neighborhood_indices, indices)
        objective_set_diff_indices = copy(objective_neighborhood_indices)
        set_nonindex_entries_to_zero!(objective_set_diff_indices.Q, set_diff_indices)
        set_nonindex_entries_to_zero!(objective_set_diff_indices.s, set_diff_indices)
        set_nonindex_entries_to_zero!(objective_set_diff_indices.grad_cst_component, set_diff_indices)
    end

    flag = false

    x = x0
    grad_indices = ∇appr_objective(objective_indices, x)
    grad_neighborhood_indices = copy(grad_indices)

    norm_grad_indices = norm(grad_indices, 2)
    d = - grad_indices
    Q_d = objective_indices.Q*d
    for t in 1:T
        eta = norm_grad_indices^2 / dot(d,Q_d)
        x .+= eta * d
        grad_indices .+= eta * Q_d
        norm_grad_indices_new = norm(grad_indices, 2)
        d = - grad_indices .+ (norm_grad_indices_new^2 / norm_grad_indices^2) * d
        Q_d = objective_indices.Q*d
        norm_grad_indices = norm_grad_indices_new

        # stop if the norm of the gradient of the objective function is less than `grad_norm_tol`
        if grad_norm_tol !== nothing && norm_grad_indices < grad_norm_tol
            break
        end

        # periodically compute the full gradient if the gradient of the objective function is non-positive for all indices in `ìndices_old`
        if length(set_diff_indices) != 0 && full_gradient_period !== nothing && t % full_gradient_period == 0 && all(grad_indices[x .> 0] .< 0)
            grad_neighborhood_indices = copy(grad_indices)
            grad_neighborhood_indices[set_diff_indices] = ∇appr_objective(objective_set_diff_indices, x)[set_diff_indices]
            if length(unique(vcat(findall(x -> x < 0, grad_neighborhood_indices), indices))) > length(indices)
                flag = true
                break
            end
        end
    end
   
    return x, grad_neighborhood_indices, flag
end


