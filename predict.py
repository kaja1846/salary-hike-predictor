import joblib

# Function to map degree input to one-hot encoding
def encode_degree(degree):
    if degree.lower() == "master's":
        return [1, 0]  # Master's degree
    elif degree.lower() == "phd":
        
        return [0, 1]  # PhD
    else:
        return [0, 0]  # Bachelor's degree

# Load the saved model
model = joblib.load('salary_hike_model.joblib')

# Prompt the user for their degree and years of experience
degree_input = input("Enter your degree (Bachelor's, Master's, PhD): ")
experience_input = float(input("Enter your years of experience: "))

# Encode the degree to match the one-hot encoded model input
degree_encoded = encode_degree(degree_input)

# Combine the years of experience with the encoded degree data
input_data = [[experience_input] + degree_encoded]

# Make a prediction using the loaded model
predicted_hike = model.predict(input_data)

# Output the predicted salary hike
print(f"Predicted Salary Hike for {degree_input} with {experience_input} years of experience: {predicted_hike[0]:.2f}%")