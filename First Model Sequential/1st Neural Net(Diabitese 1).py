# Sample Code
# Diabetese DataSet(Simple model to predict accuracy of dataset)

#dependencies
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset=loadtxt('Dataset 1.csv', delimiter=',')

#split dataset into x and y component by selecting subset of column
X = dataset[:,0:8]
y = dataset[:,8]

#defining model
'''
The model expects rows of data with 8 variables (the input_dim=8 argument)
The first hidden layer has 12 nodes and uses the relu activation function.
The second hidden layer has 8 nodes and uses the relu activation function.
The output layer has one node and uses the sigmoid activation function.
'''
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compiling the model or layers
'''
loss-loss_function, or value which one want to reduce in program using optimizer
otimizer-algorithm to update bias or probability as weight  
metrics-to check accuracy of program
'''
#binary_crossentropy is used because it is  binary classificatio n problem
#acurracy is used to show accuracy of classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=1500, batch_size=5)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

#As we increase no of epochs the accuracy too increases but it cost much time to train , however recent implementation of gpus can fix it
# also training batch size too matters:)

 





