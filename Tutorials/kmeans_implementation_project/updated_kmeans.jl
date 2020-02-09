# Replace python environment to suit your needs
ENV["PYTHON"] = "/Users/mysterio/miniconda3/envs/pydata/bin/python"
Pkg.build("PyCall")  # Build PyCall to suit the specified Python env

using PyCall
using Statistics
using LinearAlgebra
using Plots
using BenchmarkTools
using Distances


"""
"""
function Kmeans(design_matrix::Array{Float64, 2}, k::Int64; max_iters::Int64=300, tol=1e-5)
    # randomly get centroids for each group
    n_row, n_col = size(design_matrix)
    rand_indices = rand(1:n_row, k)
    centroids = design_matrix[rand_indices, :]

    # Update centroids & labels with closest members until convergence
    for iter = 1:max_iters
        Nothing
    end

    return centroids

end

Kmeans(X, 3)
