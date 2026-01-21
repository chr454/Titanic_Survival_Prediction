from flask import Flask, render_template, request
import logging
import joblib
import pandas as pd





#Initiate Flask app
app = Flask(__name__)

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

#Load the trained Pipeline
try:
  logger.info("Loading model Pipeline...")
  model = joblib.load("model/titanic_survival_model.pkl")
  logger.info("model loaded successfully")
except Exception as e:
  logger.error(f"error loading model: {e}")
  model = None

@app.route('/')
def home():
  """Renders the main page with the input form"""
  return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
  """Handles form submission, processed data, and returns prediction"""

  if not model:
    return render_template('index.html', error = "model not loaded. Please contact Administrator")
  
  try:
    # 1. Extract data from form
    # We must cast inputs to float, as HTML forms send strings
    form_data = request.form

    # 2. Create a DataFrame matching the training data structure exactly
    # The pipeline requires these specific column names
    input_data = pd.DataFrame({
      "Pclass":[int(form_data["Pclass"])],
      "Age":[int(form_data["Age"])],
      "SibSp":[int(form_data["SibSp"])],
      "Parch":[int(form_data["Parch"])],
      "Fare":[float(form_data["Fare"])],
      "Sex":[form_data["Sex"]],
      "Embarked":[form_data["Embarked"]]
    })

    Prediction = model.predict(input_data)[0]

    #return results
    result = "Survived" if Prediction == 1 else "Did not Survive"
  
    return render_template('index.html', prediction = result, input_data = form_data)
  
  except Exception as e:
    return render_template("index.html", error = str(e))
  
if __name__ == "__main__":
  app.run(debug=True)
