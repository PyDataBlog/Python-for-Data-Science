# Replace python environment to suit your needs
ENV["PYTHON"] = "/home/mysterio/miniconda3/envs/pydata/bin/python"
Pkg.build("PyCall")  # Build PyCall to suit the specified Python env

using PyCall
using Plots
using LinearAlgebra
using BenchmarkTools
using Distances



# import sklearn datasets
data = pyimport("sklearn.datasets")

X, y = data.make_blobs(n_samples=1000000, n_features=3, centers=3, cluster_std=0.9, random_state=80)


ran_k = 3
ran_x = randn(100, ran_k)
ran_l = rand(1:ran_k, 100)
ran_c = randn(ran_k, ran_k)


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


sum_of_squares(ran_x, ran_l, ran_c, ran_k)




"""
"""
function Kmeans(design_matrix::Array{Float64, 2}, k::Int64; max_iters::Int64=300, tol=1e-5, verbose::Bool=true)

    # randomly get centroids for each group
    n_row, n_col = size(design_matrix)
    rand_indices = rand(1:n_row, k)
    centroids = design_matrix[rand_indices, :]
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
            # TODO: Calculate the sum of squares
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


Kmeans(X, 3)


