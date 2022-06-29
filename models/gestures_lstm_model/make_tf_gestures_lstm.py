import tensorflow.python.keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import numpy as np
from tensorflow.lite.python.lite import TFLiteConverter
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

model = Sequential()
model.add(Input(shape=(30, 42), name="input"))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax', name="result"))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

run_model = tf.function(lambda x: model(x))

data = np.array([[[0.1 for _ in range(42)] for _ in range(30)]])
# data = np.array([[0.1 for _ in range(42)]])
result = np.array([[0, 1, 0, 0, 0, 0, 0]], float)

model.fit(data, result)
model.summary()

test_data = np.array([[[5 for _ in range(42)] for _ in range(30)]])
vals = model.predict(test_data)

# BATCHES = 1
# STEPS = 30
# INPUT_SIZE = 42
# concrete_func = run_model.get_concrete_function(
#     tf.TensorSpec([BATCHES, STEPS, INPUT_SIZE], model.inputs[0].dtype))

# concrete_func = run_model.get_concrete_function(
#     tf.TensorSpec([BATCHES, INPUT_SIZE], model.inputs[0].dtype))

# # model directory.
MODEL_DIR = "gesture_recognition_lstm"
tf.saved_model.save(model, MODEL_DIR)

# frozen_func = convert_variables_to_constants_v2(concrete_func)
# frozen_func.graph.as_graph_def()

# converter = TFLiteConverter.from_saved_model(MODEL_DIR)
# tflite_model = converter.convert()
#
# open("test.tflite", "wb").write(tflite_model)

# Save frozen graph from frozen ConcreteFunction to hard drive
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir="./",
#                   name="gesture_recognition_lstm_tf.pb",
#                   as_text=False)
