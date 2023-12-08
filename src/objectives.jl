using LinearAlgebra, Arpack, SparseArrays
import Base.copy
include("projections.jl")

abstract type AbstractObjective end

"""
The `APPRObjective` struct represents the objective function for approximate personalized PageRank. The objective has the following form:
    appr_objective(x) = 1/2 <x, Q x> - alpha <s,D_neg x> + alpha rho <ones, D_pos x>,
where Q = D_neg(D - (1-alpha)/2(D + A))D_neg, D_neg = D^{-1/2}, D_pos = D^{1/2}.

# Fields:
- `Q::SparseMatrixCSC{Float64, Int64}`: The matrix representing the quadratic term of the objective function.
- `s::SparseVector{Float64, Int64}`: The seed vector.
- `L::Float64`: The smoothness constant of the objective.
- `mu::Float64`: The strong convexity constant of the objective.
- `grad_cst_component::SparseVector{Float64, Int64}`: The constant component of the gradient of the objective function.
- ``grad_norm_0::Float64``: A bound on the norm of the gradient of the objective function at the origin.


# Functions:
- `copy(obj::APPRObjective)`: Create a copy of the objective function.
- `restrict!(obj::APPRObjective, indices::AbstractVector{Int64}, indices_child)`: Restrict the objective function to the given indices.
- `extend!(obj::APPRObjective, obj_parent::APPRObjective, indices::AbstractVector{Int64}, indices_parent::AbstractVector{Int64}` Extend the objective function with the given indices.
- `appr_objective(obj::APPRObjective, x::SparseVector{Float64, Int64}) -> Float64`: Evaluate the objective function for a given sparse vector x.
- `∇appr_objective_all(obj::APPRObjective, x::SparseVector{Float64, Int64})`: Evaluate the gradient of the objective function restricted to indices for a given sparse vector x, and return the gradient of the constant, non-constant, and complete gradient separately.
- `∇appr_objective(obj::APPRObjective, x::SparseVector{Float64, Int64})`: Evaluate the gradient of the objective function restricted to `indices`.
- `∇appr_objective_cst(obj::APPRObjective)`: Evaluate the gradient of the constant component of the objective function restricted to `indices`.
- `∇appr_objective_non_cst(obj::APPRObjective, x::SparseVector{Float64, Int64})`: Evaluate the gradient of the non-constant component of the objective function restricted to `indices`.
"""
mutable struct APPRObjective <: AbstractObjective
    Q::SparseMatrixCSC{Float64, Int64}
    s::SparseVector{Float64, Int64}
    L::Float64
    mu::Float64
    grad_cst_component::SparseVector{Float64, Int64}
    grad_norm_0::Float64
end


function copy(obj::APPRObjective)
    return APPRObjective(copy(obj.Q), copy(obj.s), copy(obj.L), copy(obj.mu), copy(obj.grad_cst_component), copy(obj.grad_norm_0))
end

function restrict!(obj::APPRObjective, indices::AbstractVector{Int64}, indices_child::AbstractVector{Int64})
    indices_diff = setdiff(indices, indices_child)
    obj.Q[indices_diff, indices] .= 0
    obj.Q[indices, indices_diff] .= 0
    obj.s[indices_diff] .= 0
    obj.grad_cst_component[indices_diff] .= 0
    dropzeros!(obj.Q)
    dropzeros!(obj.s)
    dropzeros!(obj.grad_cst_component)
end

function extend!(obj::APPRObjective, obj_parent::APPRObjective, indices::AbstractVector{Int64}, indices_parent::AbstractVector{Int64})
    indices_diff = setdiff(indices_parent, indices)
    obj.Q[indices_diff, indices_parent] = obj_parent.Q[indices_diff, indices_parent]
    obj.Q[indices_parent, indices_diff] = obj_parent.Q[indices_parent, indices_diff]
    obj.s[indices_diff] = obj_parent.s[indices_diff]
    obj.grad_cst_component[indices_diff] = obj_parent.grad_cst_component[indices_diff]
    dropzeros!(obj.Q)
    dropzeros!(obj.s)
    dropzeros!(obj.grad_cst_component)
end

function appr_objective(obj::APPRObjective, x::SparseVector{Float64, Int64})
    return 1/2*(x'*obj.Q*x) + dot(obj.grad_cst_component, x)
end

function ∇appr_objective_all(obj::APPRObjective, x::SparseVector{Float64, Int64})
     grad_cst = ∇appr_objective_cst(obj)
     grad_non_cst = ∇appr_objective_non_cst(obj, x)
     return grad_cst, grad_non_cst, grad_non_cst .+ grad_cst 
end

function ∇appr_objective(obj::APPRObjective, x::SparseVector{Float64, Int64})
    return ∇appr_objective_non_cst(obj, x) .+ ∇appr_objective_cst(obj)
end

function ∇appr_objective_cst(obj::APPRObjective)
    return dropzeros!(obj.grad_cst_component)
end

function ∇appr_objective_non_cst(obj::APPRObjective, x::SparseVector{Float64, Int64})
    return dropzeros!(obj.Q*x)
end




