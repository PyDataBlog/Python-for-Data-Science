# Replace python environment to suit your needs
ENV["PYTHON"] = "/home/mysterio/miniconda3/envs/pydata/bin/python"
Pkg.build("PyCall")  # Build PyCall to suit the specified Python env

using PyCall
using Statistics
using Plots

# import whatever
data = pyimport("sklearn.datasets")

#
X, y = data.make_blobs(n_samples=5000, n_features=3, centers=3, cluster_std=0.9, random_state=10)

# Visualize the feature space
scatter3d(X[:, 1], X[:, 2], X[:, 3], color=y, legend=false, st=:surface,
 xlabel="Feature #1", ylabel="Feature #2", zlabel="Feature #3",
 title="3D View Of Feature Space Coloured By Assigned Cluster")
