import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

class LSTM_Model(tf.keras.Model):
    """Simple stacked LSTM model for sequence-to-sequence regression.

    The model expects input shape (batch, timesteps, features) and outputs
    a vector of length `target_len` (the predicted future steps).
    """

    def __init__(self, num_layers: int = 2, units: int = 256, dropout: float = 0.1, target_len: int = 5, l2_rate:float=0.001, **kwargs):

        super().__init__(**kwargs)
        

        self.num_layers = num_layers
        self.units = units
        self.dropout = dropout
        self.target_len = target_len
        self.l2_rate = l2_rate
        l2_reg = regularizers.l2(self.l2_rate) if self.l2_rate > 0 else None

        self.lstm_layers = []
        for i in range(num_layers - 1):
            # return sequences for intermediate layers
            self.lstm_layers.append(layers.LSTM(
                units, 
                return_sequences=True,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg
            ))
        # last LSTM layer returns last output
        self.lstm_layers.append(layers.LSTM(
            units, 
            return_sequences=False,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg
        ))

        self.dropout_layer = layers.Dropout(dropout)
        self.fc1 = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)
        self.fc2 = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)
        self.out = layers.Dense(target_len)

    def call(self, x, training=False):
        for lstm in self.lstm_layers:
            x = lstm(x)
        x = self.dropout_layer(x, training=training)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "units": self.units,
            "dropout": self.dropout,
            "target_len": self.target_len,
            "l2_rate": self.l2_rate
        })
        return config


def build_lstm_model(input_shape, num_layers=2, units=128, dropout=0.2, target_len=5, l2_rate=0.001):
    """Utility to build and compile a LSTM_Model instance.

    Args:
        input_shape: tuple (timesteps, features)
    """
    model = LSTM_Model(num_layers=num_layers, units=units, dropout=dropout, target_len=target_len, l2_rate=l2_rate)

    model.build((None, input_shape[0], input_shape[1]))
    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])
    return model


__all__ = ['LSTM_Model', 'build_lstm_model']