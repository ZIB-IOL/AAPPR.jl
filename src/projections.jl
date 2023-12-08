"""
    projection_onto_orthant(x::SparseVector{Float64, Int64}; indices::Union{Nothing, Vector{Any}, Vector{Int64}} = nothing, anchor::SparseVector{Float64, Int64} = spzeros(length(x)))

Project a vector onto C = anchor + (span {e_i : i in indices} ∩ R^n_{≥ 0}). If `indices` is not specified, we project onto C = anchor + (span {e_i : i in 1:n} ∩ R^n_{≥ 0}).

# Arguments
- `x::SparseVector{Float64, Int64}`: The vector to project.
- `indices::Union{Nothing, Vector{Any}, Vector{Int64}} = nothing`: A collection of indices to project onto. By default, `indices` is `nothing` and the projection is done onto all indices of `x`.
- `anchor::SparseVector{Float64, Int64} = spzeros(length(x))`: A sparse vector at which point `C` is anchored. Default is the zero vector of the same size as `x`.

# Returns
- `result::SparseVector{Float64, Int64}`: The projection of the input vector onto the set `C`.
"""
function projection_onto_orthant(x::SparseVector{Float64, Int64}; indices::Union{Nothing, Vector{Any}, Vector{Int64}}=nothing, anchor::SparseVector{Float64, Int64}=spzeros(length(x)))
    y = dropzeros!(spzeros(length(x)))
    if indices === nothing
        y .= max.(x, anchor)
    else
        
        y[indices] .= max.(x[indices], anchor[indices])
    end

    
    return dropzeros!(y)
end


"""
    set_nonindex_entries_to_zero(x::SparseVector{Float64}, indices::Union{Vector{Any}, Vector{Int64}})

Set all entries in `x` that are not in `indices` to zero.

# Arguments
- `x::SparseVector{Float64}`: Input sparse vector to be modified.
- `indices::Union{Vector{Any}, Vector{Int64}}`: Indices to keep in `x`, other indices will be set to zero.

# Returns
- `y::SparseVector{Float64}`: Modified input vector with all entries not in `indices` set to zero.
"""
function set_nonindex_entries_to_zero(x::SparseVector{Float64}, indices::Union{Vector{Any}, Vector{Int64}})
    y = spzeros(length(x))
    y[indices] = x[indices]
    return dropzeros!(y)
end

"""
    set_nonindex_entries_to_zero!(x::SparseVector{Float64}, indices::Union{Vector{Any}, Vector{Int64}})

Set all entries in `x` that are not in `indices` to zero.

# Arguments
- `x::SparseVector{Float64}`: Input sparse vector to be modified in-place.
- `indices::Union{Vector{Any}, Vector{Int64}}`: Indices to keep in `x`, other indices will be set to zero.
"""
function set_nonindex_entries_to_zero!(x::SparseVector{Float64}, indices::Union{Vector{Any}, Vector{Int64}})
    x[setdiff(1:length(x), indices)] .= 0.0
    dropzeros!(x)
end


"""
    set_nonindex_entries_to_zero(A::SparseMatrixCSC{Float64, Int64}, indices::Union{Vector{Any}, Vector{Int64}})

Returns a copy of `A` with all entries A_{i,j} set to zero if i or j is not in `indices`.

# Arguments
- `A::SparseMatrixCSC{Float64, Int64}`: Input sparse matrix to be modified.
- `indices::Union{Vector{Any}, Vector{Int64}}`: Indices to keep in `A`, other indices will be set to zero.

# Returns
- `B::SparseMatrixCSC{Float64, Int64}`: Modified sparse matrix with entries set to zero if i or j is not in `indices`.
"""
function set_nonindex_entries_to_zero(A::SparseMatrixCSC{Float64, Int64}, indices::Union{Vector{Any}, Vector{Int64}})
    B = spzeros(Float64, size(A))
    B[indices, indices] = A[indices, indices]
    return dropzeros!(B)
end


"""
    set_nonindex_entries_to_zero!(A::SparseMatrixCSC{Float64, Int64}, indices::Union{Vector{Any}, Vector{Int64}})

This function modifies the input sparse matrix A by setting all entries A_{i,j} to zero if either i or j is not present in the indices vector.
# Arguments
- `A::SparseMatrixCSC{Float64, Int64}`: Input sparse matrix to be modified. in-place.
- `indices::Union{Vector{Any}, Vector{Int64}}`: Indices to keep in `A`, other indices will be set to zero.
"""

function set_nonindex_entries_to_zero!(A::SparseMatrixCSC{Float64, Int64}, indices::Union{Vector{Any}, Vector{Int64}})
    set_diff_A_1 = setdiff(1:size(A, 1), indices)
    set_diff_A_2 = setdiff(1:size(A, 2), indices)
    A[set_diff_A_1, :] = spzeros(Float64, length(set_diff_A_1), size(A, 2))
    A[:, set_diff_A_2] = spzeros(Float64, size(A, 1), length(set_diff_A_2))
end


"""
    reduce_and_clip(x::SparseVector{Float64}, delta::Float64)

Reduce the value of each entry of a sparse vector `x` by delta and set all negative entries to zero.
This function creates a new sparse vector `y`, which is a modified copy of `x`, and returns it.

# Arguments
- `x::SparseVector{Float64}`: Input sparse vector to be modified.
- `delta::Float64`: The value to subtract from each non-zero entry of `x`.

# Returns
- The modified sparse vector `y`.
"""
function reduce_and_clip(x::SparseVector{Float64}, delta::Float64)
    y = copy(dropzeros!(x))
    y[y .> 0] .-= delta
    y[y .< 0] .= 0
    return dropzeros!(y)
end


"""
    reduce_and_clip!(x::SparseVector{Float64}, delta::Float64)

Reduce the value of each entry of a sparse vector `x` by delta and set all negative entries to zero.

# Arguments
- `x::SparseVector{Float64}`: Input sparse vector to be modified in-place.
- `delta::Float64`: The value to subtract from each non-zero entry of `x`.

"""
function reduce_and_clip!(x::SparseVector{Float64}, delta::Float64)
    y = copy(dropzeros(x))
    x[y .> 0] .-= delta
    x[y .< 0] .= 0
end
