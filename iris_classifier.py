from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# Optional: view dataset as a DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = [target_names[i] for i in y]
print("Sample Data:")
print(df.head())

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, predictions))

# Show predictions with flower names
print("\nSample Predictions (Actual → Predicted):")
for actual, predicted in zip(y_test[:10], predictions[:10]):  # Only show first 10 for brevity
    print(f"{target_names[actual]} → {target_names[predicted]}")

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Data Scatter Plot")
plt.show()
