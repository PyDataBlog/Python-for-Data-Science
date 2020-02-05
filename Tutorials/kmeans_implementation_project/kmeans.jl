# Replace python environment to suit your needs
ENV["PYTHON"] = "/home/mysterio/miniconda3/envs/pydata/bin/python"
Pkg.build("PyCall")  # Build PyCall to suit the specified Python env

using PyCall
using Statistics
using LinearAlgebra
using Plots

# import whatever
data = pyimport("sklearn.datasets")

X, y = data.make_blobs(n_samples=100000, n_features=3, centers=3, cluster_std=0.9, random_state=80)


# Visualize the feature space
 scatter3d(X[:, 1], X[:, 2], X[:, 3], legend=false,
  xlabel="Feature #1", ylabel="Feature #2", zlabel="Feature #3",
  title="3D View Of The Feature Space", titlefontsize=11)

# Visualize the feature space using the given labels
scatter3d(X[:, 1], X[:, 2], X[:, 3], color=y, legend=false,
 xlabel="Feature #1", ylabel="Feature #2", zlabel="Feature #3",
 title="3D View Of The Feature Space Coloured By Assigned Cluster",
 titlefontsize=11)

X_list = collect(eachrow(X))


"""
"""
function init_kmeans(X, k; max_iters = 100, tol = 1e-5)
    # Reshape 2D design matrix as a list of 1D arrays
    X = collect(eachrow(X))

    # Get some info on the data being passed
    N = length(X)
    n = length(X[1])

    # Initialize some parameters based on the info
    distances = zeros(N)
    # Initiate a representative vector for each cluster (centroids)
    reps = [zeros(n) for i in 1:k]
    # randomly assign cluster to each example
    assignment = [rand(1:k) for i in 1:N]

    grp_list = []

    # rep j representative is average of points in cluster j.
    for j in 1:k
        # An array representing the index of each assigned label
        groups = [i for i in 1:N if assignment[i] == j];

        push!(grp_list, groups) # just a convinient way of populating grp_list with groups

        # Update the initialized reps array with the average of points in 
        reps[j] = sum(X[groups]) / length(groups);
    end

    return reps, assignment, grp_list

end


init_centroids, init_labels, init_group_idx = init_kmeans(X, 3)


# Visualize the feature space
 scatter3d(X[:, 1], X[:, 2], X[:, 3], legend=false, color=init_labels,
  xlabel="Feature #1", ylabel="Feature #2", zlabel="Feature #3",
  title="3D View Of The Feature Space Initialized")

scatter3d!(init_centroids[1], init_centroids[2], init_centroids[3],
 markershape=:star4, markersize=15,color="red")

    
