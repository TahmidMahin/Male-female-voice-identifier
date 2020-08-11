import pandas as pd
import numpy as np
import tensorflow as tf

path = "voice.csv"
df = pd.read_csv(path)
data = df.iloc[:, :-1].to_numpy()

gender = df.label.tolist()
label = np.zeros(len(gender))
for pos in range(len(gender)):
    if gender[pos] == "female":
        label[pos] = 1

train_data, test_data = data[:(data.shape[0]*8)//10], data[(data.shape[0]*8)//10:]
train_label, test_label = label[:(data.shape[0]*8)//10], label[(data.shape[0]*8)//10:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(train_data, train_label, epochs=50)
print("test data accuracy:")
model.evaluate(test_data, test_label)

