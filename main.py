from flask import Flask, render_template, request
import pickle
import yaml
import os
import pandas as pd
from training import Train
from sklearn.preprocessing import LabelEncoder, StandardScaler
app = Flask(__name__)

# Load configuration
if os.path.exists('input/config.yaml'):
    with open('input/config.yaml', 'r') as f:
        configs = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config file not found")

# Initialize training object
train_instance = Train(configs)
model_pipeline, x_train, x_test, y_train, y_test = train_instance.train_model()

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            text = request.form.get('text')

            if not text :
                return render_template("index.html", error="Text and day must be provided!")

            cleaned_text = train_instance.remove_tags(text)
            processed_text = train_instance.clean_text([cleaned_text])[0]

            input_data = pd.DataFrame({'text': [processed_text]})

            prediction = model_pipeline.predict(input_data)
            sentiment = train_instance.label_encoder.inverse_transform(prediction)[0]

            return render_template("result.html", text=text, sentiment=sentiment)

        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            return render_template("index.html", error=error_message)

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
