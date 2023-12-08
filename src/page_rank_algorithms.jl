using SparseArrays

include("../src/optimization_algorithms.jl")
include("../src/objectives.jl")
include("../src/graphs.jl")
include("../src/projections.jl")


"""
    accelerated_sparse_page_rank_algorithm(objective::APPRObjective; epsilon::Float64 = 0.0001, use_anchor::Bool = false, full_gradient_period::Union{Int64, Nothing} = nothing, with_nnzs_and_losses::Bool = false)

Compute an epsilon-approximate personalized PageRank vector, that is, `objective(`x`)` - `objective`(x*) ≤ `epsilon`, where x* is the optimal solution.


# Arguments
- `objective::APPRObjective`: An `APPRObjective` struct that contains the problem data.
- `epsilon::Float64 = 0.0001`: A tolerance parameter used to control the accuracy of the solution. A smaller value of `epsilon` will result in a more accurate solution, at the cost of increased computational time.
- `full_gradient_period::Union{Int64, Nothing} = nothing`: Periodicity of full gradient computation. If set to `nothing`, full gradient is not computed.
- `use_anchor::Bool = false`: If set to `true`, uses an anchor point for the projection oracle for improved efficiency. If set to `false`, does not use an anchor point. No effect on :cg.
- `variant::Symbol = :apgd`: The optimization algorithm to use. Currently, :apgd and :cg are supported. If :cg is selected, turns the algorithm in conjugate-gradients ASPR (CASPR).
- `with_nnzs_and_losses::Bool = false`: If `true`, then the nnzs and loss at each iteration will be computed and returned.

# Returns
- `x::AbstractSparseVector{Float64}`: A sparse vector containing the final solution to the approximate personalized PageRank problem.
- `times_nnzs_and_losses::Tuple{Array{Float64, 1}, Array{Int64, 1}}`: A tuple containing the times, nnzs_iterate, nnzs_neighborhood, and losses at each iteration.
"""
function accelerated_sparse_page_rank_algorithm(
    objective::APPRObjective;
    epsilon::Float64 = 0.0001,
    full_gradient_period::Union{Int64, Nothing} = nothing,
    use_anchor::Bool = false,
    variant::Symbol = :apgd,
    with_nnzs_and_losses::Bool = false
    )
    times = []
    nnzs_iterate = []
    nnzs_neighborhood = []
    losses = []
    time_start = time()
    time_losses = 0 # time lost during calculation of losses
    L = objective.L
    mu = objective.mu
    kappa = L/mu
    @assert epsilon > 0 "epsilon must be positive"
    n = length(objective.s)
    x = spzeros(n)

    # Initialize indices
    old_good_indices = Int64[]
    good_indices = Int64[]
    old_neighborhood_indices = Int64[]
    neighborhood_indices = Int64[]

    # Initialize objectives
    objective_good_indices = APPRObjective(dropzeros!(spzeros(Float64, size(objective.Q))), dropzeros!(spzeros(n)), copy(L), copy(mu), dropzeros!(spzeros(n)), copy(objective.grad_norm_0))
    objective_neighborhood_indices = copy(objective_good_indices)

    # Determine the first potentially good indices
    first_potentially_good_indices = sort(unique(findall(!iszero, objective.s)))
    extend!(objective_neighborhood_indices, objective, old_good_indices, first_potentially_good_indices)

    # Initialize the gradient
    grad = ∇appr_objective(objective_neighborhood_indices, x)

    # Determine the first good indices
    good_indices = sort(unique(findall(x -> x < 0, grad)))
    set_nonindex_entries_to_zero!(grad, good_indices)

    # Now that the good_indices have been identified, we can determine the neighborhood of the good_indices
    old_neighborhood_indices = copy(good_indices)
    neighborhood_indices = get_neighborhood_indices(good_indices, objective.Q)
    restrict!(objective_neighborhood_indices, first_potentially_good_indices, old_neighborhood_indices)

    if with_nnzs_and_losses
        time_loss = time()
        push!(nnzs_iterate, nnz(x))
        push!(nnzs_neighborhood, length(neighborhood_indices))
        push!(losses, appr_objective(objective, x))
        time_losses += time() - time_loss
        push!(times, time() - time_start - time_losses)
    end

    while good_indices != old_good_indices
        
        # Update objectives
        extend!(objective_good_indices, objective_neighborhood_indices, old_good_indices, good_indices)
        extend!(objective_neighborhood_indices, objective, old_neighborhood_indices, neighborhood_indices)
        
        delta = sqrt((epsilon * mu) / ((1 + length(good_indices))*L^2))
        grad_norm_tol = mu*delta
        
        # Set up projection oracle depending on anchor usage
        function proj(y)
            if use_anchor
                return projection_onto_orthant(y; indices=good_indices, anchor=x)
            else
                return projection_onto_orthant(y; indices=good_indices)
            end
        end

        # Call either APGD or CG depending on variant
        if variant == :apgd
            gamma = delta^2*mu/2
            T = Int64(1 + ceil(2 * sqrt(kappa) * log(((L - mu)*norm(grad, 2)^2) / (2*gamma*mu^2))))
            x, grad, flag = accelerated_projected_gradient_descent(
                objective_good_indices,
                x,
                proj,
                T;
                grad_norm_tol=grad_norm_tol,
                full_gradient_period=full_gradient_period,
                objective_neighborhood_indices=objective_neighborhood_indices,
                indices=good_indices,
                neighborhood_indices=neighborhood_indices
                )
        elseif variant == :cg
            x, grad, flag = conjugate_gradients(
                objective_good_indices,
                spzeros(n),
                n;
                grad_norm_tol=grad_norm_tol,
                full_gradient_period=full_gradient_period,
                objective_neighborhood_indices=objective_neighborhood_indices,
                indices=good_indices,
                neighborhood_indices=neighborhood_indices
            )
        end
        if flag == false
            reduce_and_clip!(x, delta)
            grad = ∇appr_objective(objective_neighborhood_indices, x)
        end

        # Update indices
        old_good_indices = copy(good_indices)
        old_neighborhood_indices = copy(neighborhood_indices)
        new_good_indices = sort(unique(setdiff(findall(x -> x < 0, grad), old_good_indices)))
        good_indices = sort(unique(vcat(new_good_indices, old_good_indices)))
        new_neighborhood_indices = get_neighborhood_indices(new_good_indices, objective.Q)
        neighborhood_indices = sort(unique(vcat(new_neighborhood_indices, old_neighborhood_indices)))

        # Update gradient
        grad = set_nonindex_entries_to_zero(grad, good_indices)

        if with_nnzs_and_losses
            time_loss = time()
            push!(nnzs_iterate, nnz(x))
            push!(nnzs_neighborhood, length(neighborhood_indices))
            push!(losses, appr_objective(objective, x))
            time_losses += time() - time_loss
            push!(times, time() - time_start - time_losses)
        end
    end
    times_nnzs_and_losses = (times, nnzs_iterate, nnzs_neighborhood, losses)
    return x, times_nnzs_and_losses
end


"""
conjugate_directions_page_rank_algorithm(objective::APPRObjective; epsilon::Float64 = 0.0001, exhaust_gradient::Bool = true, with_nnzs_and_losses::Bool = false)

Compute an epsilon-approximate personalized PageRank vector with conjugate directions (CD), that is, `objective(`x`)` - `objective`(x*) ≤ `epsilon`, where x* is the optimal solution.


# Arguments
- `objective::APPRObjective`: An `APPRObjective` struct that contains the problem data.
- `epsilon::Float64 = 0.0001`: A positive number representing the tolerance for stopping the algorithm. The default value is `0.0001`.
- `exhaust_gradient::Bool = true`: Whether or not to go through all indices corresponding to negative gradient entries before recomputing the gradient. The default value is `true`.
- `times_nnzs_and_losses::Tuple{Array{Float64, 1}, Array{Int64, 1}}`: A tuple containing the times, nnzs, and losses of the solution at each iteration.

# Returns
- `x::AbstractSparseVector{Float64}`: A sparse vector containing the final solution to the approximate personalized PageRank problem.
- `times_nnzs_and_losses::Tuple{Array{Float64, 1}, Array{Int64, 1}}`: A tuple containing the times, nnzs_iterate, nnzs_neighborhood, and losses at each iteration.
"""
function conjugate_directions_page_rank_algorithm(objective::APPRObjective; epsilon::Float64 = 0.0001, exhaust_gradient::Bool = true, with_nnzs_and_losses::Bool = false)
    iteration = 0
    times = []
    nnzs_iterate = []
    nnzs_neighborhood = []
    losses = []
    time_start = time()
    time_losses = 0 # time lost during calculation of losses
    L = objective.L
    mu = objective.mu
    @assert epsilon > 0 "epsilon must be positive"

    n = length(objective.s)
    x = spzeros(n)

    # Initialize indices
    old_neighborhood_indices = Int64[]
    neighborhood_indices = Int64[]
    incorporated_good_indices = Int64[]
    unicorporated_good_indices = Int64[]
    good_indices = Int64[]

    # Initialize objectives
    objective_incorporated_good_indices = APPRObjective(dropzeros!(spzeros(Float64, size(objective.Q))), dropzeros!(spzeros(n)), copy(L), copy(mu), dropzeros!(spzeros(n)), copy(objective.grad_norm_0))
    objective_neighborhood_indices = copy(objective_incorporated_good_indices)

    # Determine the first potentially good indices
    first_potentially_good_indices = sort(unique(findall(!iszero, objective.s)))
    extend!(objective_neighborhood_indices, objective, old_neighborhood_indices, first_potentially_good_indices)

    # Initialize the gradient
    grad = ∇appr_objective(objective_neighborhood_indices, x)

    # Determine the first good indices
    unicorporated_good_indices = sort(unique(findall(x -> x < 0, grad)))
    good_indices = copy(unicorporated_good_indices)
    set_nonindex_entries_to_zero!(grad, good_indices)

    # Now that the good_indices have been identified, we define the first neighborhood to be all of the good indices
    old_neighborhood_indices = copy(neighborhood_indices)
    neighborhood_indices = sort(unique(vcat(get_neighborhood_indices(unicorporated_good_indices, objective.Q), old_neighborhood_indices)))
    extend!(objective_neighborhood_indices, objective, old_neighborhood_indices, neighborhood_indices)

    grad_norm_tol = mu*sqrt((epsilon * mu) / ((1 + length(good_indices))*L^2))

    d_vectors = []
    Q_d_bar_vectors = []

    if with_nnzs_and_losses
        time_loss = time()
        push!(nnzs_iterate, nnz(x))
        push!(nnzs_neighborhood, length(neighborhood_indices))
        push!(losses, appr_objective(objective, x))
        time_losses += time() - time_loss
        push!(times, time() - time_start - time_losses)
    end

    while length(unicorporated_good_indices) != 0 && norm(grad, 2) ≥ grad_norm_tol
        

        if !exhaust_gradient 
            _, minindex = findmin(grad)
            unicorporated_good_indices = [minindex]
        end
        
        for i in 1:length(unicorporated_good_indices)
            idx = unicorporated_good_indices[i]
            iteration += 1
            tmp_indices = copy(incorporated_good_indices)
            push!(incorporated_good_indices, idx)
            extend!(objective_incorporated_good_indices, objective_neighborhood_indices, tmp_indices, incorporated_good_indices)
            grad_indices = spzeros(n)

            # We can save some computation time by not computing the gradient when we already have it
            if i == 1
                grad_indices[idx] = grad[idx]
            else
                grad_indices[idx] = ∇appr_objective(objective_incorporated_good_indices, x)[idx]
            end

            u = sparsevec([idx], [Float64(grad_indices[idx])], n)
            d = copy(u)
            if iteration > 1
                d .+= sum([(- dot(u, Q_d_bar_vectors[k])) * d_vectors[k] for k in 1:iteration-1])
            end
            Q_d = dropzeros(objective.Q * d)
            d_Q_d = dot(d, Q_d)
            d_bar = d / d_Q_d
            push!(d_vectors, d)
            push!(Q_d_bar_vectors, Q_d / d_Q_d)
            x .-= dot(grad_indices, d_bar) * d
        end
        

        # Update indices and objectives
        old_neighborhood_indices = copy(neighborhood_indices)
        neighborhood_indices = sort(unique(vcat(get_neighborhood_indices(unicorporated_good_indices, objective.Q), old_neighborhood_indices)))
        extend!(objective_neighborhood_indices, objective, old_neighborhood_indices, neighborhood_indices)
        grad = ∇appr_objective(objective_neighborhood_indices, x)
        unicorporated_good_indices = sort(unique(setdiff(findall(x -> x < 0, grad), incorporated_good_indices)))
        good_indices = sort(unique(vcat(unicorporated_good_indices, incorporated_good_indices)))

        # Update gradient
        set_nonindex_entries_to_zero!(grad, good_indices)
        grad_norm_tol = mu*sqrt((epsilon * mu) / ((1 + length(good_indices))*L^2))

        if with_nnzs_and_losses
            time_loss = time()
            push!(nnzs_iterate, nnz(x))
            push!(nnzs_neighborhood, length(neighborhood_indices))
            push!(losses, appr_objective(objective, x))
            time_losses += time() - time_loss
            push!(times, time() - time_start - time_losses)
        end
    end
    times_nnzs_and_losses = (times, nnzs_iterate, nnzs_neighborhood, losses)
    return x, times_nnzs_and_losses
end







"""
    ista(objective::APPRObjective; epsilon::Float64 = 0.0001, with_nnzs_and_losses::Bool = false)
    # Update indices
    old_good_indices = copy(good_indices)
    old_neighborhood_indices = copy(neighborhood_indices)
    new_good_indices = sort(unique(setdiff(findall(x -> x < 0, grad), old_good_indices)))
    good_indices = sort(unique(vcat(new_good_indices, old_good_indices)))
    new_neighborhood_indices = get_neighborhood_indices(new_good_indices, objective.Q)
    neighborhood_indices = sort(unique(vcat(new_neighborhood_indices, old_neighborhood_indices)))
Compute an epsilon-approximate personalized PageRank vector with the iterative shrinkage-thresholding algorithm (ISTA), that is, `objective(`x`)` - `objective`(x*) ≤ `epsilon`, where x* is the optimal solution. 
(We implement the algorithm with projected gradient descent, which is equivalent to ISTA for this problem.)

# Arguments
- `objective::APPRObjective`: An `APPRObjective` struct that contains the problem data.
- `epsilon::Float64 = 0.0001`: A tolerance parameter used to control the accuracy of the solution. A smaller value of `epsilon` will result in a more accurate solution, at the cost of increased computational time.
- `with_nnzs_and_losses::Bool = false`: If `true`, then the nnzs and loss at each iteration will be computed and returned.

# Returns
- `x::AbstractSparseVector{Float64}`: A sparse vector containing the final solution to the approximate personalized PageRank problem.
- `times_nnzs_and_losses::Tuple{Array{Float64, 1}, Array{Int64, 1}}`: A tuple containing the times, nnzs_iterate, nnzs_neighborhood, and losses at each iteration.

# References
- [1] Kimon Fountoulakis, Farbod Roosta-Khorasani, Julian Shun, Xiang Cheng, and Michael W Mahoney. “Variational perspective on local graph clustering”. In: Mathematical Programming 174.1 (2019), pp. 553–573
"""
function ista(objective::APPRObjective; epsilon::Float64 = 0.0001, with_nnzs_and_losses::Bool = false)
    times = []
    nnzs_iterate = []
    nnzs_neighborhood = []
    losses = []
    time_start = time()
    time_losses = 0 # time lost during calculation of losses
    L = objective.L
    mu = objective.mu
    @assert epsilon > 0 "epsilon must be positive"
    n = length(objective.s)
    x = spzeros(n)

    # Initialize indices
    old_good_indices = Int64[]
    good_indices = Int64[]
    old_neighborhood_indices = Int64[]
    neighborhood_indices = Int64[]

    # Initialize objective_good_indices and objective_neighborhood_indices
    objective_good_indices = APPRObjective(dropzeros!(spzeros(Float64, size(objective.Q))), dropzeros!(spzeros(n)), copy(L), copy(mu), dropzeros!(spzeros(n)), copy(objective.grad_norm_0))
    objective_neighborhood_indices = copy(objective_good_indices)

    # Determine the first potentially good indices
    first_potentially_good_indices = sort(unique(findall(!iszero, objective.s)))
    extend!(objective_neighborhood_indices, objective, old_good_indices, first_potentially_good_indices)

    # Initialize the gradient
    grad = ∇appr_objective(objective_neighborhood_indices, x)

    # Determine the first good indices
    good_indices = sort(unique(findall(x -> x < 0, grad)))
    set_nonindex_entries_to_zero!(grad, good_indices)

    # Now that the good_indices have been identified, we can determine the neighborhood of the good_indices
    old_neighborhood_indices = copy(good_indices)
    neighborhood_indices = get_neighborhood_indices(good_indices, objective.Q)
    restrict!(objective_neighborhood_indices, first_potentially_good_indices, old_neighborhood_indices)


    if with_nnzs_and_losses
        time_loss = time()
        push!(nnzs_iterate, nnz(x))
        push!(nnzs_neighborhood, length(neighborhood_indices))
        push!(losses, appr_objective(objective, x))
        time_losses += time() - time_loss
        push!(times, time() - time_start - time_losses)
    end

    grad_norm_tol = mu*sqrt((epsilon * mu) / ((1 + length(good_indices))*L^2))

    while norm(grad, 2) ≥ grad_norm_tol

        # Update objectives
        extend!(objective_good_indices, objective_neighborhood_indices, old_good_indices, good_indices)
        extend!(objective_neighborhood_indices, objective, old_neighborhood_indices, neighborhood_indices)

        # Perform one step of ista
        x .-= 1/L*grad
        grad = ∇appr_objective(objective_neighborhood_indices, x)

        # Update indices
        old_good_indices = copy(good_indices)
        old_neighborhood_indices = copy(neighborhood_indices)
        new_good_indices = sort(unique(setdiff(findall(x -> x < 0, grad), old_good_indices)))
        good_indices = sort(unique(vcat(new_good_indices, old_good_indices)))
        new_neighborhood_indices = get_neighborhood_indices(new_good_indices, objective.Q)
        neighborhood_indices = sort(unique(vcat(new_neighborhood_indices, old_neighborhood_indices)))

        # Update gradient
        set_nonindex_entries_to_zero!(grad, good_indices)
        grad_norm_tol = mu*sqrt((epsilon * mu) / ((1 + length(good_indices))*L^2))

        if with_nnzs_and_losses
            time_loss = time()
            push!(nnzs_iterate, nnz(x))
            push!(nnzs_neighborhood, length(neighborhood_indices))
            push!(losses, appr_objective(objective, x))
            time_losses += time() - time_loss
            push!(times, time() - time_start - time_losses)
        end
    end
    times_nnzs_and_losses = (times, nnzs_iterate, nnzs_neighborhood, losses)
    return x, times_nnzs_and_losses
end



"""
    fista(objective::APPRObjective; epsilon::Float64 = 0.0001, with_nnzs_and_losses::Bool = false)

Compute an epsilon-approximate personalized PageRank vector with the fast iterative shrinkage-thresholding algorithm (FISTA), that is, `objective(`x`)` - `objective`(x*) ≤ `epsilon`, where x* is the optimal solution. 
(We implement the algorithm with accelerated projected gradient descent, which is equivalent to FISTA for this problem.)

# Arguments
- `objective::APPRObjective`: An `APPRObjective` struct that contains the problem data.
- `epsilon::Float64 = 0.0001`: A tolerance parameter used to control the accuracy of the solution. A smaller value of `epsilon` will result in a more accurate solution, at the cost of increased computational time.
- `with_nnzs_and_losses::Bool = false`: If `true`, then the nnzs and loss at each iteration will be computed and returned.

# Returns
- `x::AbstractSparseVector{Float64}`: A sparse vector containing the final solution to the approximate personalized PageRank problem.
- `times_nnzs_and_losses::Tuple{Array{Float64, 1}, Array{Int64, 1}}`: A tuple containing the times, nnzs_iterate, nnzs_neighborhood, and losses at each iteration.

# References
- [1] Kimon Fountoulakis, Farbod Roosta-Khorasani, Julian Shun, Xiang Cheng, and Michael W Mahoney. “Variational perspective on local graph clustering”. In: Mathematical Programming 174.1 (2019), pp. 553–573
"""
function fista(objective::APPRObjective; epsilon::Float64 = 0.0001, with_nnzs_and_losses::Bool = false)
    times = []
    nnzs_iterate = []
    nnzs_neighborhood = []
    losses = []
    time_start = time()
    time_losses = 0 # time lost during calculation of losses
    L = objective.L
    mu = objective.mu
    kappa = L/mu
    @assert epsilon > 0 "epsilon must be positive"
    n = length(objective.s)
    grad_norm_0 = objective.grad_norm_0
    x = spzeros(n)
    z = copy(x)
    y = copy(z)
    A_old = 0
    a = 1
    iteration = 0
    T = 1 + 2*sqrt(kappa)*log((L-mu)*grad_norm_0^2/(2*epsilon*mu^2))

    # Initialize indices
    old_good_indices = Int64[]
    good_indices = Int64[]
    old_neighborhood_indices = Int64[]
    neighborhood_indices = Int64[]

    # Initialize objective_good_indices and objective_neighborhood_indices
    objective_good_indices = APPRObjective(dropzeros!(spzeros(Float64, size(objective.Q))), dropzeros!(spzeros(n)), copy(L), copy(mu), dropzeros!(spzeros(n)), copy(grad_norm_0))
    objective_neighborhood_indices = copy(objective_good_indices)

    # Determine the first potentially good indices
    first_potentially_good_indices = sort(unique(findall(!iszero, objective.s)))
    extend!(objective_neighborhood_indices, objective, old_good_indices, first_potentially_good_indices)

    # Initialize the gradient
    grad = ∇appr_objective(objective_neighborhood_indices, y)

    # Determine the first good indices
    good_indices = sort(unique(findall(y -> y < 0, grad)))
    set_nonindex_entries_to_zero!(grad, good_indices)

    # Now that the good_indices have been identified, we can determine the neighborhood of the good_indices
    old_neighborhood_indices = copy(good_indices)
    neighborhood_indices = get_neighborhood_indices(good_indices, objective.Q)
    restrict!(objective_neighborhood_indices, first_potentially_good_indices, old_neighborhood_indices)

    if with_nnzs_and_losses
        time_loss = time()
        push!(nnzs_iterate, nnz(y))
        push!(nnzs_neighborhood, length(neighborhood_indices))
        push!(losses, appr_objective(objective, y))
        time_losses += time() - time_loss
        push!(times, time() - time_start - time_losses)
    end

    grad_norm_tol = mu*sqrt((epsilon * mu) / ((1 + length(good_indices))*L^2))
    while norm(grad, 2) ≥ grad_norm_tol && iteration <= T
        iteration += 1
        
        # Update objectives
        extend!(objective_good_indices, objective_neighborhood_indices, old_good_indices, good_indices)
        extend!(objective_neighborhood_indices, objective, old_neighborhood_indices, neighborhood_indices)

        # Perform one step of fista
        A_new = A_old + a

        @assert A_new != NaN "A_new is NaN "
        x = (A_old/A_new)*y .+ (a/A_new)*z
        z = projection_onto_orthant((kappa-1+A_old)/(kappa-1+A_new)*z .+ (a/(kappa-1+A_new))*(x.-(1/mu)*∇appr_objective(objective_good_indices, x)))
        y = (A_old/A_new)*y .+ (a/A_new)*z
        a = A_new*(((2*kappa)/(2*kappa+1-sqrt(1+4*kappa)))-1)
        A_old = A_new

        grad = ∇appr_objective(objective_neighborhood_indices, y)

        # Update indices
        old_good_indices = copy(good_indices)
        old_neighborhood_indices = copy(neighborhood_indices)
        new_good_indices = sort(unique(setdiff(findall(x -> x < 0, grad), old_good_indices)))
        good_indices = sort(unique(vcat(new_good_indices, old_good_indices)))
        new_neighborhood_indices = get_neighborhood_indices(new_good_indices, objective.Q)
        neighborhood_indices = sort(unique(vcat(new_neighborhood_indices, old_neighborhood_indices)))

        # Update gradient
        set_nonindex_entries_to_zero!(grad, good_indices)
        grad_norm_tol = mu*sqrt((epsilon * mu) / ((1 + length(good_indices))*L^2))

        if with_nnzs_and_losses
            time_loss = time()
            push!(nnzs_iterate, nnz(y))
            push!(nnzs_neighborhood, length(neighborhood_indices))
            push!(losses, appr_objective(objective, y))
            time_losses += time() - time_loss
            push!(times, time() - time_start - time_losses)
        end
    end
    times_nnzs_and_losses = (times, nnzs_iterate, nnzs_neighborhood, losses)
    return y, times_nnzs_and_losses
end
