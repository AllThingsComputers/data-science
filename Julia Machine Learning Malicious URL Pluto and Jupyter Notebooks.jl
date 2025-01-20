# Required packages
using Pkg
Pkg.add("Conda")
Pkg.add("DataFrames")
Pkg.add("PyCall")
Pkg.add("CSV")
Pkg.add("ScikitLearn")

# Load libraries
using Conda
using DataFrames
using PyCall
using ScikitLearn
using CSV
using ScikitLearn: @sk_import

println("Libraries loaded successfully.\n")

# Load the dataset
# Replace with the full path to your dataset
df = CSV.read("C:/Users/User.user/source/repos/53 Al with more in word doc/Julia Machine Learning Malicious URL/urldata.csv", DataFrame)

# Check if the data loaded successfully
println("Dataset size: ", size(df), "\n")
println("Sample data:\n", first(df, 5)) # Print the first 5 rows

# Extract features and labels
X = convert(Array, df[!,:url])  # URLs as features
y = convert(Array, df[!,:label])  # Labels

# Create a list of URLs for vectorization
url_list = Vector{String}()
for i in df[!,:url]
    push!(url_list, i)
end

println("Sample URL from list: ", url_list[5])
println("URLs loaded successfully.\n")

# Text Vectorization using TfidfVectorizer
@sk_import feature_extraction.text: TfidfVectorizer
vectorizer = TfidfVectorizer()

println("Performing vectorization...")
Xf = fit_transform!(vectorizer, url_list)
println("Vectorization complete.")

# Split the data into training and testing sets
@sk_import model_selection: train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2, random_state=42)

println("Data split complete.")
println("Training set size: ", size(X_train))
println("Testing set size: ", size(X_test))

# Train a Logistic Regression model
@sk_import linear_model: LogisticRegression
model = LogisticRegression(fit_intercept=true)

println("Training the model...")
fit!(model, X_train, y_train)
println("Model training complete.")

# Evaluate the model
y_pred = predict(model, X_test)
accuracy = sum(y_pred .== y_test) / length(y_test)
println("Model accuracy: ", accuracy * 100, "%")
