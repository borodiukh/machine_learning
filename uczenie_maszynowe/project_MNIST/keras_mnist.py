import tensorflow as tf
from tensorflow import keras


# load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# reshape and normalize the image.
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# create the model and add layers
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
     tf.keras.layers.Dense(10, activation='softmax')
    ]
)


# add loss, optimizer and metrics
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              metrics=['accuracy'])

# train
training = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# evaluate
answer = model.evaluate(x_test, y_test)
print('Test loss:', answer[0])
print('Test accuracy:', answer[1])