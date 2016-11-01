"""
	 Some practice on the Iris dataset.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
import seaborn as sns
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn import datasets

iris = sns.load_dataset("iris")
#sns.pairplot(iris, hue='species')
#sns.plt.show()
#sns.jointplot("sepal_length", "sepal_width", iris, kind='reg' )

features = iris.values[:, 0:4]
species  = iris.values[:, 4]

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    #print("uniques -> ", uniques)
    #print("ids -> ", ids)
    return np_utils.to_categorical(ids, len(uniques))

train_X, test_X, train_y, test_y = train_test_split(features, species, train_size=0.5, random_state=0)
#species_ohe  = one_hot_encode_object_array(species)
train_y_ohe  = one_hot_encode_object_array(train_y)
test_y_ohe  = one_hot_encode_object_array(test_y)
#print(species_ohe)
#print(test_X)
#print(test_y_ohe)

model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics = ["accuracy"], optimizer='adam') # metrics = ["accuracy"]
#model.fit(features, species_ohe, validation_split=0.20, nb_epoch=150, batch_size=1)
model.fit(train_X, train_y_ohe, validation_split=0.20, nb_epoch=150, batch_size=1)
loss, accuracy = model.evaluate(test_X, test_y_ohe, verbose=0)
print("Test fraction correct (Loss) = {:.2f}".format(loss))
#print("Test fraction correct (Loss) = {:.2f}".format(loss[0]))
print("Test fraction correct (Accuracy) = {:.2f}".format(accuracy))
