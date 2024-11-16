# Импорт необходимых библиотек
import numpy as np
import tensorflow as tf

# Класс для создания наборов данных для обучения
class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None, batch_size=32):

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

    # Метод для создания датасета
    def make_dataset(self, data, input_size, output_size, batch_size=32):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=input_size + output_size,
            sequence_stride=1, shuffle=True, batch_size=batch_size)
        ds = ds.map(lambda x: self.slide_window(x, input_size, output_size))
        return ds

    # Метод для выделения входных данных и целевых признаков (labels)
    def slide_window(self, x, input_size, output_size):
        inputs = x[:, :input_size, :]
        labels = x[:, input_size:input_size + output_size, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        return inputs, labels

    @property
    def train(self):
        return self.make_dataset(self.train_df, self.input_width, self.label_width)

    @property
    def val(self):
        return self.make_dataset(self.val_df, self.input_width, self.label_width)

    @property
    def test(self):
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