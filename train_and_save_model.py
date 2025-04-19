from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple classifier
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'model.pkl')

print("✅ Model saved to model.pkl")