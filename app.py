# Импорт необходимых библиотек
from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from model import WindowGenerator, compile_and_fit
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/auto_train', methods=['POST'])
def train_model():
    # Получаем данные из запроса
    data = request.get_json()

    # Проверяем наличие обучающих данных, валидационных данных и тестовых данных
    if not all(key in data for key in ('train_data', 'val_data', 'test_data')):
        return jsonify({'error': 'Train, validation and test data are required'}), 400

    # Преобразуем входные данные в DataFrame
    train_df = pd.DataFrame(data['train_data'])
    val_df = pd.DataFrame(data['val_data'])
    test_df = pd.DataFrame(data['test_data'])

    # Настройка параметров окна
    input_width = data.get('input_width', 40)
    label_width = data.get('label_width', 144 - input_width)  # 144 - 40
    shift = data.get('shift',  144 - input_width)
    batch_size = data.get('batch_size', 32)

    # Создаем экземпляр WindowGenerator
    window = WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df)

    # Настройка гиперпараметров
    lstm_units = data.get('lstm_units', 30)
    learning_rate = data.get('learning_rate', 0.001)
    epochs = data.get('epochs', 1000)


    # Определяем модель
    num_features = train_df.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(30, return_sequences=False),
        tf.keras.layers.Dense(label_width * num_features, kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([label_width, num_features])
    ])

    # Обучаем модель
    history = compile_and_fit(model, window, learning_rate, epochs)

    # Сохраняем модель
    model_path = data.get('model_path', 'model.h5')
    model.save(model_path)

    return jsonify({'message': 'Model trained successfully', 'model_path': model_path}), 200


if __name__ == '__main__':
    app.run(port=5001)