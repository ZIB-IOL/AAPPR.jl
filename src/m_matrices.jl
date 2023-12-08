using Arpack, LinearAlgebra, SparseArrays


"""
    is_m_matrix(A::AbstractMatrix{T}) where {T<:Real}

Check if a given dense matrix is an M-matrix.

A symmetric M-matrix is a square matrix that has the following properties:
1. All diagonal entries are positive.
2. All off-diagonal entries are non-positive.
3. The matrix is symmetric.
4. The matrix is invertible, and all its eigenvalues have positive real parts.

# Arguments
- `A::AbstractMatrix{T}`: The matrix to check.

# Returns
`true` if the matrix is an M-matrix, `false` otherwise.
"""
function is_m_matrix(A::AbstractMatrix{T}) where {T<:Real}
    n = size(A, 1)
    
    # Check that the matrix is square
    if size(A, 1) != size(A, 2)
        return false
    end
    
    # Check that all diagonal entries are positive
    if any(diag(A) .<= 0)
        return false
    end
    
    # Check that all off-diagonal entries are non-positive
    if any(A[i,j] > 0 for i in 1:n, j in 1:n if i != j)
        return false
    end
    
    # Check that the matrix is symmetric
    if !isapprox(A, A', rtol=1e-8, atol=1e-8)
        return false
    end
    
    # Check that all eigenvalues have positive real parts
    if minimum(eigvals(A)) <= 0
        return false
    end
    
    # If all conditions are met, the matrix is a symmetric M-matrix
    return true
end

"""
    is_m_matrix(A::AbstractSparseMatrix{T}) where {T<:Real}

Check if a given sparse matrix is an M-matrix.

A symmetric M-matrix is a square matrix that has the following properties:
1. All diagonal entries are positive.
2. All off-diagonal entries are non-positive.
3. The matrix is symmetric.
4. The matrix is invertible, and all its eigenvalues have positive real parts.

# Arguments
- `A::AbstractSparseMatrix{T}`: The sparse matrix to check.

# Returns
`true` if the matrix is an M-matrix, `false` otherwise.
"""
function is_m_matrix(A::AbstractSparseMatrix{T}) where {T<:Real}
    n = size(A, 1)
    
    # Check that the matrix is square
    if size(A, 1) != size(A, 2)
        return false
    end
    
    # Check that all diagonal entries are positive
    if any(diag(A) .<= 0)
        return false
    end
    
    # Check that all off-diagonal entries are non-positive
    if any(A[i,j] > 0 for i in 1:n, j in 1:n if i != j)
        return false
    end
    
    # Check that the matrix is symmetric
    if !isapprox(A, A', rtol=1e-8, atol=1e-8)
        return false
    end
    
    # Check that all eigenvalues have positive real parts
    λ, _ = eigs(A, nev=1, which=:SR)
    # if minimum(real(eigs(A, 1, which=:LR))) <= 0
    if real(λ[1]) <= 0
        return false
    end
    
    # If all conditions are met, the matrix is a symmetric M-matrix
    return true
end