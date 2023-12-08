# In these file, we demonstrate how to use the algorithms in this repository to do local clustering.

# Imports
include("src/datasets.jl")
include("src/optimization_algorithms.jl")
include("src/objectives.jl")
include("src/page_rank_algorithms.jl")

# Either, one can load a graph from loadsnap64.jl, or load a graph from a file. If loading a graph from a file, make sure
# that the graph is of the type SimpleGraph{Int64} as defined in the Graphs package.
# Here, we demonstrate the pipeline using the facebook_combined graph:

(g, A, D) = loadsnap64(:facebook_combined) # get the graph, the adjacency matrix, and the degree matrix

# Next, one has to set up the distribution s, which is a vector of length n, where n is the number of vertices in the graph.
# Here, we are just going to set s to be a vector of all zeros, except for a single 1 at the index of the minimum degree vertex.
_, min_ind = findmin(D)
n = nv(g)
s = spzeros(n)
s[min_ind[1]] = 1.

# We set α = 10^(-2), ρ = 10^(-5), and ϵ = 10^(-6).
alpha = 10.0^(-2)
rho = 10.0^(-5)
epsilon = 10.0^(-6)

# Next, we set up the objective for the algorithm. 
L = 1
D_neg = D^(-1/2)
D_pos = D^(1/2)
Q = Symmetric(D_neg*(D - ((1-alpha)/2) * (D + A))*D_neg)
mu = alpha
grad_cst_component = -alpha*D_neg*s .+ (alpha*rho)*diag(D_pos)

# This is an upper bound on the norm of the gradient of the objective at the origin. Only used for FISTA.
grad_norm_0 =  norm(-alpha*D_neg*s) + alpha*rho*n^(3/2) 
obj = APPRObjective(Q, s, L, mu, grad_cst_component, grad_norm_0)

# We run all the algorithms.

# ASPR
x_aspr, _ = accelerated_sparse_page_rank_algorithm(obj; epsilon=epsilon, variant=:apgd)
# CASPR
x_caspr, _ = accelerated_sparse_page_rank_algorithm(obj; epsilon=epsilon, variant=:cg)
# CDPR
x_cdpr, _ = conjugate_directions_page_rank_algorithm(obj; epsilon=epsilon)
# FISTA
x_fista, _ = fista(obj; epsilon=epsilon)
# ISTA
x_ista, _ = ista(obj; epsilon=epsilon)

# All of these outputs will be ϵ-approximate solutions to the local clustering problem as defined in the paper.
# We print the objective value and number of nonzero entries for each of the solutions:

println("------------------------------------------------------------")
println("ASPR objective value: ", appr_objective(obj, x_aspr))
println("ASPR number of nonzero entries: ", nnz(x_aspr))
println("------------------------------------------------------------")
println("CASPR objective value: ", appr_objective(obj, x_caspr))
println("CASPR number of nonzero entries: ", nnz(x_caspr))
println("------------------------------------------------------------")
println("CDPR objective value: ", appr_objective(obj, x_cdpr))
println("CDPR number of nonzero entries: ", nnz(x_cdpr))
println("------------------------------------------------------------")
println("FISTA objective value: ", appr_objective(obj, x_fista))
println("FISTA number of nonzero entries: ", nnz(x_fista))
println("------------------------------------------------------------")
println("ISTA objective value: ", appr_objective(obj, x_ista))
println("ISTA number of nonzero entries: ", nnz(x_ista))
println("------------------------------------------------------------")


    
    