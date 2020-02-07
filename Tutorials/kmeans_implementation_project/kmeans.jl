# Replace python environment to suit your needs
ENV["PYTHON"] = "/home/mysterio/miniconda3/envs/pydata/bin/python"
Pkg.build("PyCall")  # Build PyCall to suit the specified Python env

using PyCall
using Statistics
using LinearAlgebra
using Plots

# import sklearn datasets
data = pyimport("sklearn.datasets")

X, y = data.make_blobs(n_samples=1000000, n_features=3, centers=3, cluster_std=0.9, random_state=80)


# Visualize the feature space
scatter3d(X[:, 1], X[:, 2], X[:, 3], legend=false,
 xlabel="Feature #1", ylabel="Feature #2", zlabel="Feature #3",
 title="3D View Of The Feature Space Initialized", titlefontsize=11)



""" Kmeans(X, k, max_iters = 300, tol = 1e-5)

    This function implements the kmeans algorithm and returns the assigned labels,
    representative centroids
"""
function Kmeans(X, k; max_iters = 300, tol = 1e-5)
    # Reshape 2D array to a 1D array with length of all training examples
    # where each example is of size (n, ) ie the new array is just a list of example array
    X_array_list = collect(eachrow(X))

    # Save some info on the incoming data
    N = length(X_array_list)  # Length of all training examples
    n = length(X_array_list[1])  # Length of a single training example
    distances = zeros(N)  # Empty vector for all training examples. Useful later

    # Step 1: Random initialization
    reps_centroids = [zeros(n) for grp = 1:k]  # Initiate centroids for each
    labels = rand(1:k, N)  # Randomly assign labels (between 1 to k) to all training examples

    J_previous = Inf

    for iter = 1:max_iters

        # Step 2: Update the representative centroids for each group
        for j = 1:k
            # get group indices for each group
            group_idx = [i for i = 1:N if labels[i] == j]

            # use group indices to locate each group
            reps_centroids[j] = mean(X_array_list[group_idx]);
        end;

        # Step 3: Update the group labels
        for i = 1:N
            # compute the distance between each example and the updated representative centroid
            nearest_rep_distance = [norm(X_array_list[i] - reps_centroids[x]) for x = 1:k]

            # update distances and label arrays with value and index of closest neighbour
            # findmin returns the min value & index location
            distances[i], labels[i] = findmin(nearest_rep_distance)
        end;

        # Step 4: Compute the clustering cost
        J = (norm(distances)^ 2) / N

        # Show progress and terminate if J stopped decreasing.
        println("Iteration ", iter, ": Jclust = ", J, ".")

        # Final Step 5: Check for convergence
        if iter > 1 && abs(J - J_previous) < (tol * J)
            # TODO: Calculate the sum of squares

            # Terminate algorithm with the assumption that K-means has converged
            return labels, reps_centroids

        elseif iter == max_iters && abs(J - J_previous) > (tol * J)
            throw(error("Failed to converge Check data and/or implementation or increase max_iterations"))
        end

        J_previous = J
    end

end


@time predicted_labels, centroids = Kmeans(X, 3)

cluster_centres = reduce(vcat, centroids')

# Visualize the feature space
scatter3d(X[:, 1], X[:, 2], X[:, 3], legend=false, color = predicted_labels,
    xlabel="Feature #1", ylabel="Feature #2", zlabel="Feature #3",
    title="3D View Of The Feature Space Colored by Predicted Class")

scatter3d!(cluster_centres[:, 1], cluster_centres[:, 2], cluster_centres[:, 3],
markershape=:star4, markersize=15, color="red",legend=false)


function sum_of_squares(x, labels, k)
    N = length(x)
    ss = 0

    for j = 1:k
        idx = [x for x = 1:N if labels[x] == j]
        group_data = x[idx]
        group_length = length(group_data)
        group_center = mean(group_data)

        println(group_center)
        
        for ex = 1:group_length
            group_distance = group_data[ex] .- group_center
            squared_distance = group_distance .^ 2
            total_squared_distance = sum(squared_distance)

            ss += total_squared_distance
        end
    end

    return ss
end

X_list = collect(eachrow(X))

@time sum_of_squares(X_list, predicted_labels, 3)
