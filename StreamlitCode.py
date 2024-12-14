import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
try:
    data = pd.read_csv("social_media_usage.csv")
except FileNotFoundError:
    print("Error: The file 'social_media_usage.csv' was not found. Please ensure the file is in the correct directory.")
    exit()

# Helper function to clean binary columns
def clean_binary_column(column, positive_value):
    return np.where(column == positive_value, 1, 0)

# Data processing
data['sm_li'] = clean_binary_column(data['web1h'], 1)  # LinkedIn usage
data['income'] = data['income'].apply(lambda x: x if 1 <= x <= 9 else np.nan)  # Valid income range
data['education'] = data['educ2'].apply(lambda x: x if 1 <= x <= 8 else np.nan)  # Valid education range
data['parent'] = clean_binary_column(data['par'], 1)  # Parent status
data['married'] = clean_binary_column(data['marital'], 1)  # Marital status
data['female'] = clean_binary_column(data['gender'], 2)  # Female gender
data['age'] = data['age'].apply(lambda x: x if x < 98 else np.nan)  # Valid age range

# Filter dataset and drop rows with missing values
features = ['sm_li', 'income', 'education', 'parent', 'married', 'female', 'age']
processed_data = data[features].dropna()

# Separate features (X) and target (y)
X = processed_data[['income', 'education', 'parent', 'married', 'female', 'age']]
y = processed_data['sm_li']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Define new individuals for prediction
individuals = pd.DataFrame({
    'income': [8, 8],
    'education': [7, 7],
    'parent': [0, 0],
    'married': [1, 1],
    'female': [1, 1],
    'age': [42, 82]
})

# Predict probabilities for LinkedIn usage
probabilities = model.predict_proba(individuals)[:, 1]

# Display results
for i, age in enumerate(individuals['age']):
    print(f"Probability of LinkedIn usage (age {age}): {probabilities[i]:.2f}")


