import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data and train models once
df = pd.read_csv("Cleaned-Data-Final.csv")
x = df.drop('output', axis=1)
y = df['output']

models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression()
}

# Train all models once
trained_models = {}
for name, model in models.items():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    trained_models[name] = model

# Use for accuracy testing (optional)
def run_ml_models():
    from sklearn.metrics import accuracy_score
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = f"{accuracy * 100:.2f}%"
    return results

# return predictions 
def predict_with_model(symptoms, age_group, contact_status, model_name):
    import numpy as np

    # Load the same dataset and features
    df = pd.read_csv("Cleaned-Data-Final.csv")
    x = df.drop('output', axis=1)
    y = df['output']

    # Create input template
    input_features = x.columns.tolist()
    input_data = dict.fromkeys(input_features, 0)

    # Set symptoms as 1 if selected
    for symptom in symptoms:
        input_data[symptom] = 1

    # Map age group to numerical encoding
    age_mapping = {
        "0-9": "Age_0-9", "10-19": "Age_10-19", "20-24": "Age_20-24",
        "25-59": "Age_25-59", "60+": "Age_60+"
    }
    input_data[age_mapping[age_group]] = 1

    # Contact status
    input_data["Contact"] = 1 if contact_status == "Yes" else (0.5 if contact_status == "Maybe" else 0)

    # Prepare final input vector
    input_vector = np.array([input_data[col] for col in input_features]).reshape(1, -1)

    # Select and train the chosen model
    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel='linear'),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression()
    }

    model = models[model_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)

    prediction = model.predict(input_vector)[0]
    return prediction
