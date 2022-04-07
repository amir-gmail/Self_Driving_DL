import numpy as np
import matplotlib.pyplot as plt
import random
import requests
from PIL import Image
import cv2
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

def prediction(x, model):
    return model.predict_classes(x)

def fc_model(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def leNet_model(num_classes):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation="relu")) #,strides=, padding='same'
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    np.random.seed(0)
    (x_train, y_train),  (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)
    print(x_train.shape[1:])

    # asset() takes argument if it is true or false, if true continue code smoothely otherwise sent an error
    assert(x_train.shape[0] == y_train.shape[0]), "The number of train images is not equal to the number of labels."
    assert (x_test.shape[0] == y_test.shape[0]), "The number of test images is not equal to the number of labels."
    assert (x_train.shape[1:] == (28, 28)), "The dimension of train images are not 28x28"
    assert (x_test.shape[1:] == (28, 28)), "The dimension of test images are not 28x28"
    num_of_samples = []
    cols = 5
    num_classes = 10
    fig, ax = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,10))
    fig.tight_layout() # prevent overlapping of plots
    for i in range(cols):
        for j in range(num_classes):
            x_selected = x_train[y_train == j]
            ax[j][i].imshow(x_selected[random.randint(0,len(x_selected-1)), :, :], cmap=plt.get_cmap("gray"))
            ax[j][i].axis("off")
            if i == 2:
                ax[j][i].set_title(str(j))
                num_of_samples.append(len(x_selected))
    plt.show()
    print(num_of_samples)
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_train = x_train/255
    x_test = x_test/255
    num_pixels = x_train.shape[1] * x_train.shape[2]

    # uncomment following part to use fully connected NN
    """x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)
    model = fc_model(num_pixels, num_classes)"""

    # uncomment following part to use CNN
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    model = leNet_model(num_classes)
    print(model.summary())
    history = model.fit(x_train, y_train, validation_split= 0.1, epochs= 15, batch_size= 400, verbose= 1, shuffle= 1)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'validation accuracy'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'validation loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.show()
    score = model.evaluate(x_test, y_test)
    print('Test score:', score[0])
    print('Accuracy:', score[1])
    # getting image from url
    url="https://www.researchgate.net/profile/Jose-Sempere-2/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png"
    response = requests.get(url, stream=True)
    print(response)
    # openning image with Image and show it
    img = Image.open(response.raw)
    plt.imshow(img)
    plt.show()
    # turn image to numpy array
    img_array = np.asarray(img)
    print(img_array.shape)
    # resize the image
    resized_img = cv2.resize(img_array, (28,28))
    print(resized_img.shape)
    # convert image to gray scale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    print(gray_img.shape)
    # plotting the gray scale image
    plt.imshow(gray_img, cmap=plt.get_cmap("gray"))
    plt.show()
    # turn the black pixels to white and vice versa
    revereted_img = cv2.bitwise_not(gray_img)
    plt.imshow(revereted_img, cmap=plt.get_cmap("gray"))
    plt.show()
    # normalizing and flatting the image
    image = revereted_img/255
    #image = image.reshape(1, x_test.shape[1] * x_test.shape[2]) # uncomment for fc model
    image = image.reshape(1, x_test.shape[1], x_test.shape[2], 1) # uncomment for leNet model
    output = prediction(image, model)
    print(output)
    layer1 = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
    layer2 = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)

    visual_layer1 = layer1.predict(image)
    visual_layer2 = layer2.predict(image)

    plt.figure(figsize=(10, 6))
    for i in range(30):
        plt.subplot(6, 5, i+1)
        plt.imshow(visual_layer1[0, :, :, i], cmap=plt.get_cmap("jet"))
        plt.axis('off')
    plt.title("layer1")
    plt.show()

    for i in range(15):
        plt.subplot(6, 5, i+1)
        plt.imshow(visual_layer2[0, :, :, i], cmap=plt.get_cmap("jet"))
        plt.axis('off')
    plt.title("layer2")
    plt.show()

if __name__ == "__main__":
    main()