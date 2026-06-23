import pandas as pd

column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(
    "../dataset/diabetes.csv",
    names=column_names
)

print(df.head())
print("\nDataset Information:\n")

print(df.info())
print("\nMissing Values:\n")

print(df.isnull().sum())
print("\nDataset Statistics:\n")

print(df.describe())
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="Outcome", data=df)

plt.title("Diabetes Distribution")

plt.show()
# Features and Target

X = df.drop("Outcome", axis=1)

y = df["Outcome"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully!")
y_pred = model.predict(X_test)

print("Predictions generated successfully!")
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))