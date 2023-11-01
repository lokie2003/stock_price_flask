from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the pre-trained LSTM model
model = tf.keras.models.load_model("Stockpriceflask.h5")
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict_stock_price():
    predicted_price = None

    if request.method == 'POST':
        try:
            inputs = []
            for i in range(10):
                input_name = f"day_{i+1}"
                input_value = float(request.form[input_name])
                inputs.append(input_value)

            # Prepare the input data for prediction
            input_data = np.array(inputs).reshape(1, 10, 1)

            # Make the prediction
            prediction = model.predict(input_data)

            # Inverse transform the prediction to get the actual stock price
            predicted_price = scaler.inverse_transform(prediction)

            # Ensure the predicted price is non-negative
            predicted_price = max(323.60, predicted_price[0][0])

        except Exception as e:
            return jsonify({"error": str(e)})

    return render_template('predict_form.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)

