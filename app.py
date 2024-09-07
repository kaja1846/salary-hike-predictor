import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Sample data creation
data = {
    'ID': [1, 2, 3, 4, 5, 6],
    'Degree': ['Bachelor\'s', 'Master\'s', 'PhD', 'Bachelor\'s', 'Master\'s', 'PhD'],
    'Experience (Years)': [1, 3, 5, 7, 10, 12],
    'Current Salary (USD)': [75000, 95000, 125000, 110000, 140000, 170000],
    'Salary Hike (%)': [5, 10, 12, 8, 15, 20]
}

df = pd.DataFrame(data)

# Display the dataset
print("Sample Data:\n", df)

# 1. Feature Engineering - Convert 'Degree' to One-Hot Encoding
encoder = OneHotEncoder(drop='first')  # Drop first to avoid multicollinearity
degree_encoded = encoder.fit_transform(df[['Degree']]).toarray()

# Combine One-Hot encoded degree data back into the DataFrame
degree_labels = encoder.categories_[0][1:]  # Skip the first category as it's dropped
degree_df = pd.DataFrame(degree_encoded, columns=degree_labels)

# Merge with the original data
df = pd.concat([df.drop('Degree', axis=1), degree_df], axis=1)

# Display the data after encoding
print("\nData after One-Hot Encoding:\n", df)

# 2. Defining Features (X) and Target (y)
X = df[['Experience (Years)', 'Master\'s', 'PhD']]  # Features
y = df['Salary Hike (%)']  # Target

# 3. Splitting the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Selection - Using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'salary_hike_model.joblib')

print("Model saved to 'salary_hike_model.joblib'")

# Load the saved model from the file
loaded_model = joblib.load('salary_hike_model.joblib')

# Now you can use the loaded model to make predictions
new_data = [[6, 1, 0]]  # Example: 6 years experience, Master's degree (1), PhD (0)
predicted_hike = loaded_model.predict(new_data)

print("Predicted Salary Hike for a Master's degree with 6 years experience: {:.2f}%".format(predicted_hike[0]))

# 5. Predict on the test set
y_pred = model.predict(X_test)

# 6. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error (MAE):", mae)
print("R-squared (R2 Score):", r2)

# Predict salary hike for a new candidate (Example: Master's Degree, 6 years of experience)
new_data = [[6, 1, 0]]  # 6 years of experience, Master's degree (1), PhD (0)
predicted_hike = model.predict(new_data)

print("\nPredicted Salary Hike for a Master's degree with 6 years experience: {:.2f}%".format(predicted_hike[0]))