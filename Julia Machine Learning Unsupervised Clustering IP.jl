# Import necessary packages
using Pkg
Pkg.add("RDatasets")
Pkg.add("Clustering")
Pkg.add("Plots")
Pkg.add("CSV")
Pkg.add("DataFrames")

using CSV
using DataFrames
using Clustering
using Plots

# Load the dataset
df = CSV.read("hits_against_Website.csv", DataFrame)
println("Dataset loaded successfully!")
println(df)

# Ensure proper column names for clustering
# Replace `1:2` with the appropriate column indices for clustering
features = collect(Matrix(df[:, 1:2])')  # Adjust column indices based on your dataset

# Perform K-means clustering with 3 clusters
result = kmeans(features, 3)

# Print clustering results
println("Clustering completed!")
println("Cluster assignments: ", result.assignments)

# Plot the result
scatter(
    df."hits", df."hour",               # Replace with actual column names in your dataset
    marker_z = result.assignments,
    color = :lightrainbow,
    legend = false,
    xlabel = "Hits",
    ylabel = "Hour",
    title = "K-means Clustering"
)

# Save the plot as a PNG file
output_path = "C:\\Users\\User.user\\source\\repos\\53 Al with more in word doc\\hits_against_Website.png"
savefig(output_path)
println("Plot saved at: ", output_path)
