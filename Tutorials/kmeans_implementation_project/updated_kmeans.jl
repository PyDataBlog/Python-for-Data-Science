using Pkg
# Replace python environment to suit your needs
ENV["PYTHON"] = "/home/mysterio/miniconda3/envs/pydata/bin/python"
Pkg.build("PyCall")  # Build PyCall to suit the specified Python env

using PyCall
using Plots
using LinearAlgebra
using Statistics
using StatsBase
using BenchmarkTools
using Distances


# import sklearn datasets
data = pyimport("sklearn.datasets")
X, y = data.make_blobs(n_samples=1000000, n_features=3, centers=3, cluster_std=0.9, random_state=80)


"""
"""
function smart_init(X::Array{Float64, 2}, k::Int; init::String="k-means++")
    n_row, n_col = size(X)

    if init == "k-means++"
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

            if i == (k-1)
                break
            end

            # compute distances from the centroids to all data points
            # and update the squared distance as the minimum distance to all centroid
            current_centroid_matrix = convert(Matrix, centroids[i, :]')
            new_distances = vec(pairwise(Euclidean(), X, current_centroid_matrix, dims = 1))

            distances = minimum([distances, new_distances])

        end

    else
        rand_indices = rand(1:n_row, k)
        centroids = X[rand_indices, :]

    end

    return centroids, n_row, n_col
end


"""
"""
function sum_of_squares(x::Array{Float64,2}, labels::Array{Int64,1}, centre::Array, k::Int)
    ss = 0

    for j = 1:k
        group_data = x[findall(labels .== j), :]
        group_centroid_matrix  = convert(Matrix, centre[j, :]')
        group_distance = pairwise(Euclidean(), group_data, group_centroid_matrix, dims=1)

        ss += sum(group_distance .^ 2)
    end

    return ss
end


"""
"""
function Kmeans(design_matrix::Array{Float64, 2}, k::Int64; k_init::String="k-means++",
    max_iters::Int64=300, tol=1e-4, verbose::Bool=true)

    centroids, n_row, n_col = smart_init(design_matrix, k, init=k_init)

    labels = rand(1:k, n_row)
    distances = zeros(n_row)

    J_previous = Inf64

    # Update centroids & labels with closest members until convergence
    for iter = 1:max_iters
        nearest_neighbour = pairwise(Euclidean(), design_matrix, centroids, dims=1)

        min_val_idx = findmin.(eachrow(nearest_neighbour))

        distances = [x[1] for x in min_val_idx]
        labels = [x[2] for x in min_val_idx]

        centroids = [ mean( X[findall(labels .== j), : ], dims = 1) for j = 1:k]
        centroids = reduce(vcat, centroids)

        J = (norm(distances)^ 2) / n_row

        if verbose
            # Show progress and terminate if J stopped decreasing.
            println("Iteration ", iter, ": Jclust = ", J, ".")
        end;

        # Final Step 5: Check for convergence
        if iter > 1 && abs(J - J_previous) < (tol * J)

            sum_squares = sum_of_squares(design_matrix, labels, centroids, k)
            # Terminate algorithm with the assumption that K-means has converged
            if verbose
                println("Successfully terminated with convergence.")
            end

            return labels, centroids, sum_squares

        elseif iter == max_iters && abs(J - J_previous) > (tol * J)
            throw(error("Failed to converge Check data and/or implementation or increase max_iter."))
        end;

        J_previous = J
    end

end


@btime begin
    num = []
    ss = []
    for i = 2:10
        l, c, s = Kmeans(X, i, k_init="k-means++", verbose=false)
        push!(num, i)
        push!(ss, s)
    end
end


plot(num, ss, ylabel="Sum of Squares", xlabel="Number of Iterations",
     title = "Test For Heterogeneity Per Iteration", legend=false)


function test_speed(x)
    for i = 2:10
        l, c, s = Kmeans(X, i, k_init="k-means++", verbose=false)
    end
end

r = @benchmark test_speed(X) samples=7 seconds=300

