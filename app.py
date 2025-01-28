# Импорт необходимых библиотек
import os
import optuna
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

from model import compile_and_fit, window_generator_create

app = Flask(__name__)
CORS(app)


@app.route('/auto_train', methods=['POST'])
def train_model():
    """
    Функция представления обучения модели
    :return: JSON типа {'message': 'Model trained successfully', 'model_path': Путь к модели}
    """
    try:
        # Проверяем, что запрос содержит данные JSON
        if not request.json:
            return jsonify({'error': 'Request must be in JSON format'}), 400

        # Получаем данные из запроса
        data = request.json

        # Проверяем обязательные поля
        required_fields = ['lstm_units', 'learning_rate', 'epochs', 'model_path']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Генерируем данные и окно
        try:
            data, train_df, label_width, window = window_generator_create()
        except Exception as e:
            return jsonify({'error': f'Error in window generator: {str(e)}'}), 500

        # Настройка гиперпараметров
        lstm_units = data['lstm_units']
        learning_rate = data['learning_rate']
        epochs = data['epochs']
        model_path = data['model_path']

        # Убедимся, что путь к модели существует
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Определяем модель
        num_features = train_df.shape[1]
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            tf.keras.layers.Dense(label_width * num_features, kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([label_width, num_features])
        ])

        # Обучаем модель
        try:
            history = compile_and_fit(model, window, learning_rate, epochs)
        except Exception as e:
            return jsonify({'error': f'Error during model training: {str(e)}'}), 500

        # Сохраняем модель
        try:
            model.save(model_path)
        except Exception as e:
            return jsonify({'error': f'Error saving model: {str(e)}'}), 500

        return jsonify({'message': 'Model trained successfully', 'model_path': model_path}), 200

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/auto_train_hyperparams', methods=['POST'])
def train_hyperparams():
    """
    Функция представления обучения модели с автоматическим подбором гиперпараметров
    :return: JSON типа {'message': 'Model trained successfully with optimal hyperparameters',
                        'model_path': Путь к модели,
                        'best_params': Перечень оптимальных гиперпараметров}
    """
    try:
        # Проверяем, что запрос содержит данные JSON
        if not request.json:
            return jsonify({'error': 'Request must be in JSON format'}), 400

        # Получаем данные из запроса
        data = request.json

        # Проверяем обязательное поле для пути модели
        if 'model_path' not in data:
            return jsonify({'error': 'Missing required field: model_path'}), 400

        model_path = data['model_path']

        # Генерируем данные и окно
        try:
            data, train_df, label_width, window = window_generator_create()
        except Exception as e:
            return jsonify({'error': f'Error in window generator: {str(e)}'}), 500

        # Проверяем, что данные корректны
        if train_df is None or label_width is None or window is None:
            return jsonify({'error': 'Generated data or window is invalid'}), 500

        # Функция оптимизации для Optuna
        def objective(trial):
            try:
                # Оптимизируемые гиперпараметры
                lstm_units = trial.suggest_int('lstm_units', 10, 100)
                learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
                epochs = trial.suggest_int('epochs', 1, 10)

                # Определяем модель
                num_features = train_df.shape[1]
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(lstm_units, return_sequences=False),
                    tf.keras.layers.Dense(label_width * num_features, kernel_initializer=tf.initializers.zeros()),
                    tf.keras.layers.Reshape([label_width, num_features])
                ])

                # Компилируем модель
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                              loss='mean_squared_error')

                # Обучаем модель
                history = model.fit(window.train, validation_data=window.val, epochs=epochs, verbose=0)

                # Возвращаем минимальную валидационную ошибку
                val_loss = min(history.history['val_loss'])
                return val_loss

            except Exception as e:
                raise RuntimeError(f'Error during trial: {str(e)}')

        # Создаем Optuna-исследование
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)  # Число попыток подбора параметров

        # Получаем лучшие параметры
        best_params = study.best_params

        # Тренируем финальную модель с лучшими параметрами
        lstm_units = best_params['lstm_units']
        learning_rate = best_params['learning_rate']
        epochs = best_params['epochs']

        num_features = train_df.shape[1]
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            tf.keras.layers.Dense(label_width * num_features, kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([label_width, num_features])
        ])

        # Компилируем и обучаем финальную модель
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])

        history = model.fit(window.train, validation_data=window.val, epochs=epochs, verbose=1)

        # Рассчитываем MAE на валидационной выборке
        val_mae = history.history['val_mean_absolute_error'][-1]

        # Убедимся, что путь к модели существует
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Сохраняем модель
        model.save(model_path)

        return jsonify({
            'message': 'Model trained successfully with optimal hyperparameters',
            'model_path': model_path,
            'best_params': best_params,
            'val_mae': val_mae
        }), 200

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5001)