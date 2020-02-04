# Replace python environment to suit your needs
ENV["PYTHON"] = "YOUR/PYTHON/ENVIRONMENT/DIRECTORY/HERE"

# Build PyCall to suit the specified Python env
Pkg.build("PyCall")  

# Import needed libraries
using PyCall
using Statistics
using Plots

# Import sklearn dataset generator
data = pyimport("sklearn.datasets")

# Generate clustered data
X, y = data.make_blobs(n_samples=5000, n_features=3, centers=3, cluster_std=0.9, random_state=10)

# Visualize the feature space
scatter3d(X[:, 1], X[:, 2], X[:, 3], color=y, legend=false, st=:surface,
 xlabel="Feature #1", ylabel="Feature #2", zlabel="Feature #3",
 title="3D View Of Feature Space Coloured By Assigned Cluster")
