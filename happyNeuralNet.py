import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model
from keras import losses
from keras.callbacks import EarlyStopping
import sys
import torch.nn as nn


filename = "stars.csv"
trainDataSet = pd.read_csv(filename, sep=',', header=None, low_memory=False)
X = trainDataSet.values[1:, 1:]
Y = trainDataSet.values[1:, 0]
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=.35)
layer_size = X.shape[1]

loss_vals = []
width_vals = []
fit_hist = []
num_models = int(sys.argv[1])

print(layer_size)

for m in range(num_models):
    width = m ** 2 + 1
    if width > 250:
        width = 249
    print('model num: ', m, ' with width: ', width)

    input_layer = Input(shape=(layer_size,))
    encode_layer0 = Dense(10000, activation='relu')(input_layer)
    encode_layer05 = Dense(5000, activation='relu')(encode_layer0)
    encode_layer1 = Dense(1000, activation='relu')(encode_layer05)
    encode_layer15 = Dense(700, activation='relu')(encode_layer1)
    encode_layer2 = Dense(500, activation='relu')(encode_layer15)
    encode_layer3 = Dense(250, activation='relu')(encode_layer2)

    bottleneck = Dense(width, activation='sigmoid')(encode_layer3)

    decode_layer1 = Dense(250, activation='relu')(bottleneck)
    decode_layer2 = Dense(500, activation='relu')(decode_layer1)
    decode_layer25 = Dense(700, activation='relu')(decode_layer1)
    decode_layer3 = Dense(1000, activation='relu')(decode_layer25)
    decode_layer35 = Dense(5000, activation='relu')(decode_layer3)
    decode_layer4 = Dense(10000, activation='relu')(decode_layer35)

    output_layer = Dense(layer_size)(decode_layer4)

    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    fit_mod = model.fit(train_x, train_x, epochs=10, batch_size=2048, validation_data=(test_x, test_x),
                        callbacks=[early_stopping])
    fit_hist.append(fit_mod.history.values())
    # loss_vals.append(min(min(fit_mod.history.values())))
    print("BOTTLENECK_WIDTH: ", bottleneck)
    loss_vals.append(np.mean(min(fit_mod.history.values())))
    width_vals.append(width)
    # model.predict(test_x)

plt.title("Loss Values as Bottleneck Width Increases Trained With " + str(num_models) + " Models")
plt.plot(width_vals, loss_vals)
plt.ylabel("Mean Loss Values")
plt.xlabel("Bottleneck Width")
plt.show()

#
# opt_loss = 1.0
# loss_val_dict = fit_hist
# for a in loss_val_dict:
#     opt_loss = min(a)
#
# feat_num = input_layer.get_shape()[1]
# if opt_loss <= 0.1:
#     feat_num = bottleneck.get_shape()[1]
#
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.fc1 = nn.Linear(in_features=192, out_features=120)
#         self.fc2 = nn.Linear(in_features=120, out_features=feat_num)
#         self.fc3 = nn.Linear(in_features=feat_num, out_features=60)
#         self.out = nn.Linear(in_features=60, out_features=10)
#
#     def forward(self, t):
#         # implement the forward pass
#         return t
#
# test_net = Network()
