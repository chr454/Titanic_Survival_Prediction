import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib



# Load the Data set
print("Loading the dataset...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Seperate Features and Target
X = train.drop(["Survived", "PassengerId","Name","Ticket","Cabin"], axis=1)
y = train["Survived"]

#Define features for Preprocessing
numerical_features = [
  "Pclass", "Age", "SibSp", "Parch", "Fare"
]
categorical_features = [
  "Sex", "Embarked"
]

#Build Preprocessing Pipeline
numeric_transformer = Pipeline(steps =[
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy="most_frequent")),
  ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

# Combine them into a single column Transformer
Preprocessor = ColumnTransformer(
  transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
  ])

# Create a full pipeline
model_pipeline = Pipeline(steps=[
  ('preprocessor', Preprocessor),
  ('Classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

#Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Pipleline
print("Training the pipeline (preprocessing + model)")
model_pipeline.fit(X_train,y_train)

# Evaluating 
print("Evaluating ...")
predictions = model_pipeline.predict(X_test)

# Metrics

accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)

print("_" * 30)
print(f"Accuracy = {accuracy:.4f} (How often is the model correct?)")
print(f"Recall = {recall:.4f} (Of all who actually survived, how many did we catch?)")
print(f"Precision = {precision:.4f} (when it predicts 'Survived', how often is it right)")

# Save the pipeline
joblib.dump(model_pipeline, 'model/titanic_survival_model.pkl')
print("model saved as titanic_survival_model.pkl")

