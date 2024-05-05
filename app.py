import numpy as np
import sqlite3
import io
import base64

# from random import randrange
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model, model_from_json


app = Flask(__name__)

db_path = 'users.db'
gan_model_path = 'model/generator_model_080.keras'
cnn_model_json_path = 'model/train.json'
cnn_model_weights_path = 'model/train.weights.h5'

dr_classes = {
    0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'
}

global gan_model
global cnn_model

gan_model = load_model(gan_model_path)
with open(cnn_model_json_path) as train_json:
    cnn_model = model_from_json(train_json.read())
cnn_model.load_weights(cnn_model_weights_path)
cnn_model.make_predict_function()


def generate_latent_data(latent_dim, n_samples):
    return (
        np.random.randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        '''SELECT * FROM users WHERE email = ? AND password = ?''',
        (email, password)
    )
    user = cursor.fetchone()
    conn.close()

    if user:
        return redirect(url_for('main_app'))
    else:
        return redirect(url_for('index'))


@app.route('/app')
def main_app():
    return render_template('main_app.html')


@app.route('/predict', methods=['POST'])
def predict():
    global gan_model
    global cnn_model

    file = base64.b64decode(eval(request.data.decode())['image'])
    query_img = np.array(Image.open(io.BytesIO(file)).resize((32, 32)))

    # x = gan_model.predict(generate_latent_data(200, 200))
    # img = np.asarray(x[randrange(200), :, :]).reshape(1, 32, 32, 3)
    img = query_img.reshape(1, 32, 32, 3)
    prediction = cnn_model.predict(img)
    prediction = np.argmax(prediction[0])
    return jsonify({'prediction': dr_classes.get(prediction, 'None')})


if __name__ == '__main__':
    app.run(debug=True)
