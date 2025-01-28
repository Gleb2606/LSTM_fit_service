# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import jsonify, request

class WindowGenerator:
    """
    Класс набор данных
    """
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None, batch_size=32):
        """
        Конструктор класса
        :param input_width: Размер входного окна
        :param label_width: Размер выходного окна
        :param shift: Сдвиг
        :param train_df: Обучающая выборка
        :param val_df: Валидационная выборка
        :param test_df: Тестовая выборка
        :param label_columns: Целевые признаки
        :param batch_size: Размер батча
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        self.batch_size = batch_size
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def make_dataset(self, data, input_size, output_size, batch_size=32):
        """
        Метод формирования датасета
        :param data: Данные
        :param input_size: Размер входного окна
        :param output_size: Размер выходного окна
        :param batch_size: Размер батча
        :return: датасет
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=input_size + output_size,
            sequence_stride=1, shuffle=True, batch_size=batch_size)
        ds = ds.map(lambda x: self.slide_window(x, input_size, output_size))
        return ds

    # Метод для выделения входных данных и целевых признаков (labels)
    def slide_window(self, x, input_size, output_size):
        """
        Метод выделения целевых признаков
        :param x: Временной ряд
        :param input_size: Размер входного окна
        :param output_size: Размер выходного окна
        :return: входы, метки
        """
        inputs = x[:, :input_size, :]
        labels = x[:, input_size:input_size + output_size, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        return inputs, labels

    @property
    def train(self):
        """
        Метод формирования обучающей выборки
        :return: Обучающая выборка
        """
        return self.make_dataset(self.train_df, self.input_width, self.label_width)

    @property
    def val(self):
        """
        Метод формирования валидационной выборки
        :return: Валидационная выборка
        """
        return self.make_dataset(self.val_df, self.input_width, self.label_width)

    @property
    def test(self):
        """
        Метод формирования тестовой выборки
        :return: Тестовая выборка
        """
        return self.make_dataset(self.test_df, self.input_width, self.label_width)

# Функция для обучения модели
def compile_and_fit(model, window, learning_rate=0.001, epochs=1500):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val, callbacks=[early_stopping])
    return history

def window_generator_create():
    """
    Функция создания окна для обучения модели
    :return: данные, обучающая выборка, окно наблюдения, окно прогноза
    """
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
    shift = data.get('shift', 144 - input_width)
    batch_size = data.get('batch_size', 32)

    # Создаем экземпляр WindowGenerator
    window = WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df)

    return data, train_df, label_width, window