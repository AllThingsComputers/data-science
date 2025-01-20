# Install required libraries (uncomment if needed)
# !pip install tensorflow
# !pip install numpy

# Import required libraries
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset (adjust delimiter if needed, e.g., ',' or ' ')
dataset = loadtxt('SecurityDataANN.csv', delimiter=' ')
print("Dataset loaded successfully!")

# Split dataset into input (X) and output (y) variables
X = dataset[:, 0:8]  # First 8 columns
y = dataset[:, 8]    # Last column (assessment: 0 or 1)

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=1)

# Evaluate the Keras model
accuracy = model.evaluate(X, y, verbose=0)
print(f'Accuracy: {accuracy[1]*100:.2f}%')

# Display the dataset structure (optional)
print("Dataset shape:", dataset.shape)

# Predictions Section
# Make predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)

# Summarize the first 15 cases
print("\nPredictions for the first 15 cases:")
for i in range(15):
    print(f"Input: {X[i].tolist()} => Predicted: {predictions[i][0]} (Expected: {int(y[i])})")

