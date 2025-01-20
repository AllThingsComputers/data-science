# Install and load the 'prophet' library (if not already installed)
if (!require(prophet)) {
  install.packages("prophet")
}
library(prophet)

# Load the dataset
# Replace the file path with the actual path to your dataset
ap <- read.csv("C:\\Users\\User.user\\source\\Monthly_Incidents.csv")

# Ensure the dataset is formatted correctly
# Prophet expects the columns to be named 'ds' (date) and 'y' (value)
colnames(ap) <- c("ds", "y") # Adjust this if your CSV has different column names
ap$ds <- as.Date(ap$ds)     # Convert 'ds' to Date format if not already

# Create a Prophet model
m <- prophet(ap)

# Create a future dataframe for predictions
# Adjust the periods (number of days) as needed
future <- make_future_dataframe(m, periods = 365)

# Print the future dataframe for inspection
cat("\nFuture Dataframe:\n")
print(tail(future))

# Generate forecasts
forecast <- predict(m, future)

# Print the last few rows of the forecast
cat("\nForecast Predictions:\n")
print(tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')]))

# Save the main forecast plot as a PNG file
png(file = "C:\\Users\\User.user\\source\\Monthly_IncidentsGFG.png")
plot(m, forecast)
dev.off()

# Save the trend and component plots as another PNG file
png(file = "C:\\Users\\User.user\\source\\Monthly_IncidentstrendGFG.png")
prophet_plot_components(m, forecast)
dev.off()

cat("\nPlots saved successfully.\n")
