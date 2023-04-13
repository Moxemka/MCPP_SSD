print('loading APIs')

import tensorflow as tf
import os
import cv2
import numpy as np
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

print('loaded :)')

warnings.filterwarnings("ignore", category=FutureWarning)

train_directory = 'train' ##directory for train data
test_directory = 'test' ##directory for test data
image_WH = 244 #image width and height
bathsize = 32 #bath size workload



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
    from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()

    for batch in test_data.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(f"Model tested. presition: {pre.result().numpy()}, recall: { re.result().numpy()}, accuracy: {acc.result().numpy()}")

##architecture of our model
def model_design(img_height, img_width):
    n_classes = count_classes(train_directory)

    conv_base  = tf.keras.applications.DenseNet201(include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3))
    for layer in conv_base.layers:
        layer.trainable = False
    
    top_model = layers.Flatten()(conv_base.output)
    top_model = layers.Dense(n_classes * 2, activation='sigmoid')(top_model)
    top_model = layers.Dropout(0.25)(top_model)
    output_layer = layers.Dense(n_classes, activation='softmax')(top_model)

    model = tf.keras.models.Model(inputs=conv_base.input, outputs=output_layer)
  

    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    
    return model


##train and test trained model
def train_N_test_model(epocs_num):

    #_train_ds = tf.keras.utils.image_dataset_from_directory(train_directory, validation_split=0.2, subset="training", label_mode='categorical', seed=123, image_size=(image_WH, image_WH), batch_size=bathsize)
    #val_ds = tf.keras.utils.image_dataset_from_directory(train_directory, validation_split=0.2, subset="validation", label_mode='categorical', seed=123, image_size=(image_WH, image_WH), batch_size=bathsize)
    test_ds = tf.keras.utils.image_dataset_from_directory(test_directory, seed=123, image_size=(image_WH, image_WH), label_mode='categorical', batch_size=bathsize)
    

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3
    )

    train_ds = train_datagen.flow_from_directory(
        train_directory,
        batch_size=bathsize,
        seed=123,
        target_size =(image_WH, image_WH),
        class_mode='categorical',
        subset='training'
    )

    image_classes = list(train_ds.class_indices.keys())

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_ds = val_datagen.flow_from_directory(
        train_directory,
        batch_size=bathsize,
        seed=123,
        target_size =(image_WH, image_WH),
        class_mode='categorical',
        subset='validation'

    )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_ds = test_datagen.flow_from_directory(
        test_directory,
        batch_size=bathsize,
        seed=123,
        target_size=(image_WH, image_WH),
        class_mode='categorical'
    ) 
    

    model = model_design(image_WH, image_WH) #initialising model for training
    

    print('awaiting for creating log...')
    #creating log (never used)
    tensor_callback = tf.keras.callbacks.TensorBoard(log_dir='log')

    #callback for early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    print('log created \nawaiting for GYM opening')

    hist = model.fit(train_ds, epochs=epocs_num, steps_per_epoch = train_ds.samples // bathsize, validation_data=val_ds, validation_steps = val_ds.samples // bathsize, callbacks=[tensor_callback, es]) ##Gym for  model
    
    

    pc_save_model(model, image_classes)

    hist_graph(hist)

    #presison_accuracy_recall(model, test_ds)
    loss, accuracy, precision, recall = model.evaluate(test_ds)
    print('Test accuracy:', accuracy)
    print('Test precision:', precision)
    print('Test recall:', recall)
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
