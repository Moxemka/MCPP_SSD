print('loading APIs')

import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

print('loaded :)')

directory = 'Planes' ##directory for data
image_WH = 500 #image width and higth
bathsize = 8 #bath size workload

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
def pc_delete_model(name='tmp'):
    os.remove(os.path.join('models', name + '_classes' + '.txt'))
    os.remove(os.path.join('models',f'{name}.h5'))

#save model automaticcaly
def pc_save_model(model, image_classes, name='tmp'):
    file_4_classes(image_classes, name)
    model.save(os.path.join('models',f'{name}.h5'))

#save model with input
def user_save_model(model, image_classes, name=1):
    print(f'save_this_model?')
    ans = input()
    if ans == "y" or ans == 'Y':
        print('Enter name of model')
        name = input()
        pc_save_model(model, image_classes, name)
        pc_delete_model()
        print("model saved")
    else:
        print('Model saved for later use: tmp.h5')

##architecture of our model
def model_design(image_h, image_w):
    model = Sequential() #CNN model
    activation = 'sigmoid'
    final_layers = count_classes(directory)
    #modeling CNN // modification from VGG model with 3 blocks + dropout + batch normalization

    model.add(Conv2D(32, 3, activation=activation, padding='same', kernel_initializer = 'he_uniform',input_shape=(image_h, image_w, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D())

    model.add(Flatten())

    # fully connected layers
    model.add(Dense(128, activation=activation,  kernel_initializer = 'he_uniform'))
    model.add(Dropout(0.05))
    model.add(Dense(64, activation=activation,  kernel_initializer = 'he_uniform'))
    model.add(Dropout(0.05))

    # final layer
    
    model.add(Dense(final_layers, activation='softmax'))

    # compiling model
    model.compile(loss='kullback_leibler_divergence', optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

    model.summary() #model info

    return model

##important graphs
def hist_graph(hist):

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
def presison_accuracy_recall(model, test_data):
    from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test_data.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(f"Model tested. presition: {pre.result().numpy()}, recall: { re.result().numpy()}, accuracy: {acc.result().numpy()}")

##train and test trained model
def train_N_test_model(epocs_num):

    image_data = tf.keras.utils.image_dataset_from_directory(directory, label_mode='categorical', batch_size=bathsize, image_size=(image_WH, image_WH)) #creating tensor data from images
    image_classes = image_data.class_names
    data_iterator = image_data.as_numpy_iterator()
    batch = data_iterator.next() #creating bath tensor
    #plt.figure(figsize=(10, 10))

    #fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    #for idx, img in enumerate(batch[0][:4]):
    #    ax[idx].imshow(img.astype(int))
    #    ax[idx].title.set_text(batch[1][idx])
    #plt.show()

    image_data = image_data.map(lambda x,y: (x/image_WH, y)) #optimising tensor from 0 -> image_WH? to 0 -> 1
    image_data.as_numpy_iterator().next() #moving to other bath tensor

    ##creating numbers of data baths for training, validation, testing
    train_data_size = int(len(image_data)*.65)
    val_data_size = int(len(image_data)*.2) 
    test_data_size = int(len(image_data)*.15) 

    ##taking images for training, validation and testing
    train = image_data.take(train_data_size)
    val = image_data.skip(train_data_size).take(val_data_size)
    test = image_data.skip(train_data_size+val_data_size).take(test_data_size)
    print(f'Test data len {len(test)}, train data len {len(train)}, val data len {len(val)}')

    model = model_design(image_WH, image_WH) #initialising model for training
    

    print('awaiting for creating log...')
    #creating log (never used)
    tensor_callback = tf.keras.callbacks.TensorBoard(log_dir='log')
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    print('log created \nawaiting for GYM opening')

    hist = model.fit(train, epochs=epocs_num, validation_data=val, callbacks=[tensor_callback, es]) ##Gym for  model
    

    pc_save_model(model, image_classes)

    hist_graph(hist)

    presison_accuracy_recall(model, test)
    
    user_save_model(model, image_classes)
    
    #image_path='229280_800.jpg'
    #print(f'prediction: {test_model(model, image_classes, image_path)}')


#testing with unknown for CNN picture
def test_model(model, image_classes, image_path):
    try:
        img = cv2.imread(image_path)
        plt.imshow(img)
        plt.show()
    except:
        return 'issue with image'
    

    #resizing 2 fit into algorithm
    resize = tf.image.resize(img, (image_WH, image_WH))
    yhat = model.predict(np.expand_dims(resize/image_WH, 0))
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
        return False
  
            
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
        while model != False:
            print('input picture path:')
            image_path = input()
            
            if image_path == 'exit':
                break
            else:
                print(test_model(model, class_names, image_path))



##in case of gpu limit memory causing OOM exeption
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
print(f'found {len(gpus)} GPUs: {gpus}')

while True:
    print('command: train or open model?')
    command = input()
    if command == "train": train_N_test_model(try_to_int_ans())
    elif command == "open": using_model()
    else: print('unknown command')
