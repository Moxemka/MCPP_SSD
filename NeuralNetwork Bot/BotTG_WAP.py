print('loading API')
import telebot as tb
import tensorflow as tf
import os
import os
import cv2
import numpy as np
import PIL
from tensorflow.keras.models import load_model
print('loaded \nloading model')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
print(f'found {len(gpus)} GPUs: {gpus}')

bot = tb.TeleBot('')
image_WH = 299
model_name = '80per'

def class_from_file(name):
    hist = []
    try:
      f = open(os.path.join('models', name + '_classes' + '.txt'), 'r')
    except Exception as e: 
        print(f"file not found, {e}") 
        _hist = [0]
        return _hist
    for line in f:
        hist.append(line)
    f.close()
    return hist

def _load_model(model_name):
    try:
       return load_model(os.path.join('models',f'{model_name}.h5'))
    except Exception as e:
        print(f'{e}')

def test_model(model, image_classes, image_path):
    try:
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    except:
        print('issue with image')

    #resizing 2 fit into algorithm
    resize = tf.image.resize(img, (image_WH, image_WH))
    yhat = model.predict(np.expand_dims(resize/image_WH, 0))
    max_yhat = yhat.argmax(axis=1)
    pred_class = (f'Это {image_classes[int(max_yhat)]}\nя уверен в этом на {int(yhat[0, int(max_yhat)] * 100)}%')
    print(pred_class)
    return pred_class




@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, 'Привет, это тг бот для определения типа самолета, просто отправьте в чат фотку')

@bot.message_handler(commands=['help'])
def main(message):
    bot.send_message(message.chat.id, 
                     'отправьте картинку для определения модели самолета\nвведите /classes чтобы узнать какие самолеты умеет распознавать эта сеть')


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        print(f'\nphoto from @{message.from_user.username}')
        filepath = 'received/'
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = filepath + message.photo[-1].file_id + '.jpg'
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        ans = test_model(model, classes, src)
        bot.reply_to(message, ans) 
        os.remove(src)
    except Exception as e:
        bot.reply_to(message, e)

@bot.message_handler(commands=['summary'])
def main(message):
    bot.send_message(message.chat.id, f'Я состою из {len(model.layers)} слоёв\n\n')

@bot.message_handler(commands=['classes'])
def main(message):
    n_class=len(classes)
    delimiter = ' '
    classese = delimiter.join(classes)
    bot.send_message(message.chat.id, 'Я умею распознавать:\n\n' + classese + f'\n всего {n_class}')


@bot.message_handler(content_types=['text'])
def main(message):
    print(f'message from @{message.from_user.username} {message.from_user.first_name}:\n{message.text}')



classes = class_from_file(model_name)
model = _load_model(model_name)
print('ready')


bot.infinity_polling()