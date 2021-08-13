import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import sys

def trainPhoneFinderRegression(dataset_path):

    txt_file = os.path.join(dataset_path, 'labels.txt')

    total_imgs = 0
    with open(txt_file, 'r') as file:
        for line in file:
            total_imgs += 1
    file.close()

    # Get size of train and test dataset
    train_sz = int(total_imgs*0.7)
    test_sz = total_imgs - train_sz

    # Divide data into train and test
    img_size = 128
    img_size_ch = 3 

    # Placeholders for train_set_X, train_set_Y and val_set_X, val_set_Y
    train_set_X = np.zeros(shape = [train_sz, img_size, img_size, img_size_ch])
    train_set_Y = np.zeros(shape = [train_sz, 2])

    val_set_X = np.zeros(shape = [test_sz, img_size, img_size, img_size_ch])
    val_set_Y = np.zeros(shape = [test_sz, 2])

    val_set = np.zeros(shape = [test_sz, img_size, img_size, img_size_ch])
    train_set = np.zeros(shape = [train_sz, img_size, img_size, img_size_ch])

    count = 0

    with open(txt_file, 'r') as file:
        for line in file:
            count += 1
            vals = line.split(' ')

            img = cv2.imread(os.path.join(dataset_path + '/' + vals[0]))
            resized_img = cv2.resize(img, (img_size, img_size))
            normalized_img = np.interp(resized_img, (resized_img.min(), resized_img.max()), (0, 1))

            if count <= train_sz:
                train_set[count-1, :, :, :] = normalized_img
                train_set_X[count-1, :, :, :] = normalized_img

                train_set_Y[count-1, 0] = float(vals[1])
                train_set_Y[count-1, 1] = float(vals[2])

            else:
                val_set[abs(train_sz-count)-1, :, :, :] = normalized_img
                val_set_X[abs(train_sz-count)-1, :, :, :] = normalized_img

                val_set_Y[abs(train_sz-count)-1, 0] = float(vals[1])
                val_set_Y[abs(train_sz-count)-1, 1] = float(vals[2])


    file.close()

    # Best model
    input = keras.layers.Input(shape = (img_size, img_size, img_size_ch), dtype='float')

    x = keras.layers.Conv2D(64, (3,3), activation='relu', strides = (1,1), padding = 'same')(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D(pool_size = (2,2))(x)
    x = keras.layers.SpatialDropout2D(0.20)(x)

    x = keras.layers.Conv2D(64, (3,3), activation='relu', strides = (2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D(pool_size = (2,2))(x)
    x = keras.layers.SpatialDropout2D(0.20)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units =128, activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dense(units =64, activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    output = keras.layers.Dense(units =2, activation = 'sigmoid', kernel_regularizer=keras.regularizers.l2(0.01))(x)

    model = keras.Model(input, output)

    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model_history = model.fit(train_set_X, train_set_Y, epochs = 100, verbose = 1, validation_data=(val_set_X, val_set_Y))
    model.save('Phone_finder_Model.h5')

    fig = plt.figure()
    fig.suptitle('Loss Plot', fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(model_history.history['loss'], label='train')
    plt.plot(model_history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('Loss_Plot.jpg')
    plt.show()

def main():
    # Parse images and labels.txt directory path to make Yolov3 annotation 
    trainPhoneFinderRegression(sys.argv[1])

if __name__ == "__main__":
    main()

