import tensorflow as tf
from tensorflow.keras import layers


class LSTM_Model(tf.keras.Model):
    """Simple stacked LSTM model for sequence-to-sequence regression.

    The model expects input shape (batch, timesteps, features) and outputs
    a vector of length `target_len` (the predicted future steps).
    """

    # 1. Thêm **kwargs
    def __init__(self, num_layers: int = 2, units: int = 128, dropout: float = 0.2, target_len: int = 5, **kwargs):
        # 2. Truyền **kwargs vào super()
        super().__init__(**kwargs)
        
        # 3. Lưu các tham số cho get_config
        self.num_layers = num_layers
        self.units = units
        self.dropout = dropout
        self.target_len = target_len

        self.lstm_layers = []
        for i in range(num_layers - 1):
            # return sequences for intermediate layers
            self.lstm_layers.append(layers.LSTM(units, return_sequences=True))
        # last LSTM layer returns last output
        self.lstm_layers.append(layers.LSTM(units, return_sequences=False))

        self.dropout_layer = layers.Dropout(dropout)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.out = layers.Dense(target_len)

    def call(self, x, training=False):
        for lstm in self.lstm_layers:
            x = lstm(x)
        x = self.dropout_layer(x, training=training)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

    # 4. Thêm get_config
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "units": self.units,
            "dropout": self.dropout,
            "target_len": self.target_len,
        })
        return config


def build_lstm_model(input_shape, num_layers=2, units=128, dropout=0.2, target_len=5):
    """Utility to build and compile a LSTM_Model instance.

    Args:
        input_shape: tuple (timesteps, features)
    """
    model = LSTM_Model(num_layers=num_layers, units=units, dropout=dropout, target_len=target_len)
    # build the model by calling it on a dummy input
    model.build((None, input_shape[0], input_shape[1]))
    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])
    return model


__all__ = ['LSTM_Model', 'build_lstm_model']