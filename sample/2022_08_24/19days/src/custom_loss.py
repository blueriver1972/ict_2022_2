import tensorflow as tf 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from tensorflow import keras 


# def huber_fn(y_true, y_pred):
#     error = y_true - y_pred
#     is_small_error = tf.abs(error) < 1
#     squared_loss = tf.square(error) / 2
#     linear_loss = tf.abs(error) - 0.5
#     return tf.where(is_small_error, squared_loss, linear_loss)

class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}
    
#TODO 1. loss function
#TODO 2. dense layer
#TODO 3. activation in layer
#TODO 4. kernel init in layer
#TODO 5. kernel regulaarizer in layer
#TODO 6. training loop alternavite model.fit

if __name__ =="__main__":
    #NOTE 1. Load Data
    housing = fetch_california_housing()
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data, housing.target.reshape(-1, 1), random_state=42
    )
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    input_shape = X_train.shape[1:]
    
    #NOTE 2. Build Model
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                        input_shape=input_shape),
        keras.layers.Dense(1, activation=my_softplus,
                        kernel_regularizer=MyL1Regularizer(0.01),
                        kernel_constraint=my_positive_weights,
                        kernel_initializer=my_glorot_initializer),
    ])
    
    #NOTE 3-1.
    #model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
    model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])
    #NOTE 3-2.
    model.fit(
        X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid)
    )