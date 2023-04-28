print('loading APIs')

import tensorflow as tf
import os
import cv2
import numpy as np
import PIL
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

print('loaded :)')


train_directory = 'planes_new_train' ##directory for train data
test_directory = 'planes_new_test - Copy' ##directory for train data
image_WH = 299 #image width and height
bathsize = 10 #bath size workload
steps_per_epoch = 100

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
def restruct_unexistant_images(path):
    quality = 90
    extension = '.jpeg'
    for root, dirs, files in os.walk(path):
        for file in files:
            # extention of file
            ext = os.path.splitext(file)[1]

            # Ext is jpg or JPEG
            # Проверяем, что расширение файла не является jpg или jpeg
            if ext.lower() != '.jpeg' or ext.lower() != '.jpg':
                img_path = os.path.join(root, file)

                # oppening file with Pillow and saving it in JPEG format
                try:
                    img = Image.open(img_path)
                    img.save(os.path.splitext(img_path)[0] + extension, 'JPEG', quality=quality)
                    print(os.path.join(root, file), 'has been converted to JPEG')
                except IOError:
                    print(os.path.join(root, file), 'cannot be converted')

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
    plt.plot(hist.history['categorical_accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_categorical_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

#evaluate presition, recall and accuacy
def presison_accuracy_recall(model, test_data):
    from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()

    for batch in test_data: 
        X, y = batch
        yhat = model.predict(X, verbose=1)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(f"Model tested. presition: {pre.result().numpy()}, recall: { re.result().numpy()}, accuracy: {acc.result().numpy()}")

##architecture of our model
def model_design(img_height, img_width):
    n_classes = count_classes(train_directory)

    conv_base  = tf.keras.applications.DenceNet201(include_top=False, 
                                                   weights="imagenet", 
                                                   input_shape=(img_height, img_width, 3))

    
    for layer in conv_base.layers:
        layer.trainable = False

    top_model = layers.GlobalAveragePooling2D()(conv_base.output)
    
    top_model = layers.Dense(1024, activation='relu')(top_model)
    top_model = layers.BatchNormalization()(top_model)
    top_model = layers.Dropout(0.5)(top_model)

    top_model = layers.Dense(512, activation='relu')(top_model)
    top_model = layers.BatchNormalization()(top_model)
    top_model = layers.Dropout(0.5)(top_model)

    output_layer = layers.Dense(n_classes, activation='softmax')(top_model)

    model = tf.keras.models.Model(inputs=conv_base.input, outputs=output_layer)
  

    model.summary()
    
    return model

def freeze_base_layers(model, frozen_per_layers=1.0):
    base_layers_count = len(model.layers)
    fine_tune_at = int(base_layers_count*frozen_per_layers)
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    print('Frozen {} lyers out of {} \n'.format(fine_tune_at, base_layers_count))
    return model

##train and test trained model
def train_N_test_model(epocs_num):
    

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range = 20,
        brightness_range=[0.5, 1.5],
        horizontal_flip = True,
        validation_split=0.2
    )

    train_ds = train_datagen.flow_from_directory(
        train_directory,
        batch_size=bathsize,
        seed=123,
        shuffle=True,
        target_size =(image_WH, image_WH),
        class_mode='categorical',
        subset='training'
    )

    image_classes = list(train_ds.class_indices.keys())

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_ds = val_datagen.flow_from_directory(
        train_directory,
        batch_size=bathsize,
        shuffle=False,
        target_size = (image_WH, image_WH),
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
    
    #___________________________stage_1

    model_1 = model_design(image_WH, image_WH) #initialising model for training
    
    opt = tf.keras.optimizers.Adamax(learning_rate=0.01)
    model_1.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['categorical_accuracy'])

    print('\nawaiting for GYM opening')

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, min_lr = 0.00001),
        tf.keras.callbacks.ModelCheckpoint(monitor='val_categorical_accuracy', mode='max', save_best_only=True, filepath=os.path.join('models',f'_best.h5'), verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=5, restore_best_weights=True)
    ]


    print('stage 1')
    hist = model_1.fit(train_ds, 
                     epochs=epocs_num, 
                     steps_per_epoch = train_ds.samples // bathsize, 
                     validation_data=val_ds, validation_steps = val_ds.samples // bathsize, 
                     callbacks=[callbacks]
                     ) ##Gym for  model
    
    pc_save_model(model_1, image_classes, 'tmp1')
    
    #__________________stage_2
    
    print('stage 2')
    model_2 = _load_model('_best')
    model_2.trainable = True        
    model_2 = freeze_base_layers(model_2, frozen_per_layers=0.75)

   
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model_2.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['categorical_accuracy'])
    

    hist = model_2.fit(train_ds, 
                     epochs=10, 
                     steps_per_epoch = train_ds.samples // bathsize, 
                     validation_data=val_ds, validation_steps = val_ds.samples // bathsize, 
                     callbacks=[callbacks]
                     ) ##Finetuning the model
    
    pc_save_model(model_2, image_classes, 'tmp2')

    #___________________________stage_3

    print('stage 3')
    model_3 = _load_model('_best')
    model_3.trainable = True      
    model_3 = freeze_base_layers(model_2, frozen_per_layers=0.5)

    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model_3.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['categorical_accuracy'])
    

    hist = model_3.fit(train_ds, 
                     epochs=15, 
                     steps_per_epoch = train_ds.samples // bathsize, 
                     validation_data=val_ds, validation_steps = val_ds.samples // bathsize, 
                     callbacks=[callbacks]
                     ) ##Finetuning the model


  
    pc_save_model(model_3, image_classes, 'tmp3')

    #_____________stage_4
    print('stage 4')
    model_4 = _load_model('_best')
    model_4.trainable = True      
    model_4 = freeze_base_layers(model_2, frozen_per_layers=0)

   
    

    opt = tf.keras.optimizers.Adamax(learning_rate=0.000001)
    model_4.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['categorical_accuracy'])
    

    hist = model_4.fit(train_ds, 
                     epochs=20, 
                     steps_per_epoch = train_ds.samples // bathsize, 
                     validation_data=val_ds, validation_steps = val_ds.samples // bathsize, 
                     callbacks=[callbacks]
                     ) ##Finetuning the model

    pc_save_model(model_4, image_classes, 'tmp4')
    
    hist_graph(hist)


    print('evaluating model')
    presison_accuracy_recall(model_4, test_ds)
    result = model_4.evaluate(test_ds)
    print(f"stage 4 Test-set classification accuracy: {result[1]}")

    user_save_model(model_4, image_classes)
    

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
    pred_class = f'this is {image_classes[int(max_yhat)]} with score {yhat[0, int(max_yhat)] * 100}%' 
    
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
