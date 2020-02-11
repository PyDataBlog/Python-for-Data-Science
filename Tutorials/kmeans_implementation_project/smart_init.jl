using Pkg
# Replace python environment to suit your needs
ENV["PYTHON"] = "/home/mysterio/miniconda3/envs/pydata/bin/python"
Pkg.build("PyCall")  # Build PyCall to suit the specified Python env

using PyCall
using Plots
using LinearAlgebra
using Statistics
using BenchmarkTools
using Distances


# import sklearn datasets
data = pyimport("sklearn.datasets")
X, y = data.make_blobs(n_samples = 300, centers = 4, random_state = 0, cluster_std = 0.6)


function smart_init(X, k, init="++")
    n_row, n_col = size(X)

    if init == "++"
        # randonmly select the first centroid from the data (X)
        centroids = zeros(k, n_col)
        rand_idx = rand(1:n_row)
        centroids[1, :] = X[rand_idx, :]

        # compute distances from the first centroid chosen to all the other data points
        first_centroid_matrix = convert(Matrix, centroids[1, :]')
        # flattened vector (n_row,)
        distances = vec(pairwise(Euclidean(), X, first_centroid_matrix, dims = 1))

        for i = 2:k
            # choose the next centroid, the probability for each data point to be chosen
            # is directly proportional to its squared distance from the nearest centroid
            prob = distances .^ 2
            r_idx = sample(1:n_row, ProbabilityWeights(prob))
            centroids[i, :] = X[r_idx, :]
        end
    end

    return centroids
end
