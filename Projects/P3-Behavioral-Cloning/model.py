import csv
import os
import cv2
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
import sklearn

DATA_ROOT = 'E:\\myproject\\big_project\\Simulator\\106_2\\'
DRIVING_LOG = os.path.join(DATA_ROOT, 'driving_log.csv')

def read_data():
    samples = []
    correction = 0.2
    with open(DRIVING_LOG) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_img_path = line[0]
            left_img_path = line[1]
            right_img_path = line[2]

            angel = float(line[3])
            center_angel = angel
            left_angel = angel + correction
            right_angel = angel - correction

            samples.append((center_img_path, center_angel))
            samples.append((left_img_path, left_angel))
            samples.append((right_img_path, right_angel))
    return samples


def flip_image(image):
    return cv2.flip(image, 1)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for item in batch_samples:
                angel = float(item[1])
                image = cv2.imread(item[0])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if random.randint(0, 1):
                    images.append(image)
                    angles.append(angel)
                else:
                    augmented_image = flip_image(image)
                    augmented_angel = angel * -1
                    images.append(augmented_image)
                    angles.append(augmented_angel)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model


samples = read_data()
random.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = build_model()
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=5)

model.save('model.h5')

# print(history_object.keys())
#
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
