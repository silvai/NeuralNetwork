import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import torch.nn as nn

trainDataSet = pd.read_csv("stars.csv", sep=',', header=None, low_memory=False)
X = trainDataSet.values[1:, 1:]
Y = trainDataSet.values[1:, 0]
cv = train_test_split(X, Y, test_size=.30)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=.30)

training_accuracy = []
test_accuracy = []
validation_accuracy = []
layer_values = range(25)  # Up to 8 hidden layers

# different number of hidden layers
for layer in layer_values:
    hiddens = tuple(layer * [16])
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddens, random_state=1)
    clf.fit(train_x, train_y)
    print('layer:', layer)
    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))
    cv = cross_val_score(clf, train_x, train_y, cv=2).mean()
    validation_accuracy.append(cv)
plt.style.use('ggplot')
fig = plt.figure()
plt.plot(layer_values, training_accuracy, 'r', label="Training Accuracy")
plt.plot(layer_values, test_accuracy, 'g', label="Testing Accuracy")
# plt.plot(layer_values, validation_accuracy, 'b', label="Cross Validation Score")
plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy')
plt.title('stars: Number of Hidden Layer\'s versus Accuracy')
plt.legend(loc='best')
plt.show()
plt.close(fig)


# different number of neurons #

neurons = range(1, 30)
for neuron in neurons:
    hiddens = 11 * [neuron]
    # clf = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=hiddens, random_state=1)
    clf = MLPClassifier.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 2))
    clf.fit(train_x, train_y)
    print('neuron:', neuron)
    training_accuracy.append(accuracy_score(train_y, clf.predict(train_x)))
    test_accuracy.append(accuracy_score(test_y, clf.predict(test_x)))
    cv = cross_val_score(clf, train_x, train_y, cv=2).mean()
    validation_accuracy.append(cv)
plt.style.use('classic')
fig = plt.figure()
plt.plot(neurons, training_accuracy, 'r', label="Training Accuracy")
plt.plot(neurons, test_accuracy, 'g', label="Testing Accuracy")
# plt.plot(layer_values, validation_accuracy, 'b', label="Cross Validation Score")
plt.xlabel('Number of Neurons')
plt.ylabel('Accuracy')
plt.title('stars: Number of Neurons\'s versus Accuracy')
plt.legend(loc='best')

plt.show()

print('____________________________________________________________________________________________________________')

x = MLPClassifier.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 2))

print(x)
