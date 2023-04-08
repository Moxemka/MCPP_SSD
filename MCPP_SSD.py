print('loading APIs')
from asyncio.windows_events import NULL
from email.mime import image
from shutil import ExecError
import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

print('loaded :)')
directory = 'Planes' ##directory for planes

##in case of gpu limit memory causing OOM exeption
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


#this function creates a file of hist with defined name
def file_4_classes(classes, name):
      f = open(os.path.join('models', name + '_classes' + '.txt'), 'w')
      for i in range(0, len(classes)):
          f.write(str(classes[i]) + '\n')
      f.close()
      print('file created with name = ',name,'.txt')

def class_from_file(name):
    hist = []
    try:
      f = open(os.path.join('models', name + '_classes' + '.txt'), 'r')
    except: 
        print("class file not found") 
        _hist = [0]
        return _hist
    for line in f:
        hist.append(line)
    f.close()
    return hist

##scan directory for images
def remove_unexistant_images(_dir):
    data_dir = _dir
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    classes = 0
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image) #creating image path
            try: #trying to open image
                img = cv2.imread(image_path) 
                extention = imghdr.what(image_path)
                if extention not in image_exts: 
                    print(f'Image not in ext list {format(image_path)}')
                    os.remove(image_path)
            except: 
                print(f'Issue with image {format(image_path)}')
                #os.remove(image_path)

#count classes(folders) in directory
def count_classes(_dir):
    return int(len(os.listdir(_dir)))

#save model
def _save_model(model, image_classes):
    print(f'save_this_model?')
    ans = input()
    if ans == "y" or ans == 'Y':
        print('Enter name of model')
        name = input()
        file_4_classes(image_classes, name)
        model.save(os.path.join('models',f'{name}.h5'))
        print("model saved")
    else:
        print('model scraped :c')

##architecture of our model
def model_design(image_h, image_w):
    model = Sequential() #CNN model

    #modeling CNN // modification from VGG-16

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_h, image_w, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    model.add(Dense(count_classes(directory), activation='softmax'))
    

    return model

##train and test trained model
def train_N_test_model(epocs_num):

    image_data = tf.keras.utils.image_dataset_from_directory(directory, label_mode='categorical', labels="inferred", batch_size=20) #creating tensor data from images
    image_classes = image_data.class_names

    data_iterator = image_data.as_numpy_iterator()
    batch = data_iterator.next() #creating bath tensor
    image_data = image_data.map(lambda x,y: (x/256, y)) #optimising tensor from 0 -> 256? to 0 -> 1
    image_data.as_numpy_iterator().next() #moving to other bath tensor

    ##creating numbers of data baths for training, validation, testing
    train_data_size = int(len(image_data)*.7)
    val_data_size = int(len(image_data)*.2)
    test_data_size = int(len(image_data)*.1) + 1

    ##taking images for training, validation and testing
    train = image_data.take(train_data_size)
    val = image_data.skip(train_data_size).take(val_data_size)
    test = image_data.skip(train_data_size+val_data_size).take(test_data_size)
    print(f'Test data len {len(test)}, train data len {len(train)}, val data len {len(val)}')

    model = model_design(256, 256) #initialising model for training
    opt = SGD(lr=0.001, momentum=0.9) #optimising with stochastic gradient descent

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) #compiling model with categorical crossentropy loss function and metering accuracy
    model.summary() #model info
    print('awaiting for creating log...')
    #creating log (never used)
    logdir='log'
    tensor_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    print('log created \n awaiting for GYM opening')

    hist = model.fit(train, epochs=epocs_num, validation_data=val, callbacks=[tensor_callback]) ##Gym for  model


    ##important graphs
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    #evaluate presition, recall and accuacy
    from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(f"presition: {pre.result().numpy()}, recall: { re.result().numpy()},accuracy: {acc.result().numpy()}")
    
    image_path='229280_800.jpg'
    print(f'prediction: {test_model(model, image_classes, image_path)}')

    _save_model(model, image_classes)


#testing with unknown for CNN picture
def test_model(model, image_classes, image_path):
    try:
        img = cv2.imread(image_path)
    except:
        print('issue with image')
        return 1
    plt.imshow(img)
    plt.show()

    #resizing 2 fit into algorithm
    resize = tf.image.resize(img, (256,256))
    yhat = model.predict(np.expand_dims(resize/256, 0))
    max_yhat = yhat.argmax(axis=1)
    pred_class = image_classes[int(max_yhat) - 1]
    
    return pred_class

#load model from directory
def _load_model(model_name):
    try:
       return load_model(os.path.join('models',f'{model_name}.h5'))
    except:
        print('class model not found, train new model instead? y/n?')
        ans = input()
        if ans == 'y' or ans == 'Y':
            train_N_test_model(try_to_int_ans())
  
            
#try to take int value
def try_to_int_ans():
    while True:
        try:
            print('Input number of epochs to train')
            ans = int(input())
            return ans
        except:
            print('Epocs must be integer value')

#open and use trained model
def using_model():
    while True:  
        print('input model name or exit to exit:')
        model_name = input()
        if model_name == 'exit':
            break
        model = _load_model(model_name)
        class_names = class_from_file(model_name)

        print('input picture path:')
        image_path = input()

        print(test_model(model, class_names, image_path))


while True:
    print('command: train or open model?')
    command = input()
    if command == "train": train_N_test_model(try_to_int_ans())
    elif command == "open": using_model()
    else: print('unknown command')
