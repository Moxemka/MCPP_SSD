import os
from PIL import Image

# Задаем путь к директории
dir_path = '_test'

# Проходим по всем имеющимся папкам
for root, dirs, files in os.walk(dir_path):
    for file in files:
        # Проверяем, является ли текущий файл изображением JPEG или JPG
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            # Открываем текущее изображение
            img = Image.open(os.path.join(root, file))
            # Получаем размеры изображения
            width, height = img.size
            # Определяем, насколько пикселей обрезать нижнюю часть изображения
            pixels_to_crop = int(height * 0.2)
            # Обрезаем нижние 20% изображения
            img = img.crop((0, 0, width, height - pixels_to_crop))
            # Сохраняем измененное изображение с тем же именем файла
            img.save(os.path.join(root, file))


dir_path = '_train'

for root, dirs, files in os.walk(dir_path):
    for file in files:
        # Проверяем, является ли текущий файл изображением JPEG или JPG
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            # Открываем текущее изображение
            img = Image.open(os.path.join(root, file))
            # Получаем размеры изображения
            width, height = img.size
            # Определяем, насколько пикселей обрезать нижнюю часть изображения
            pixels_to_crop = int(height * 0.2)
            # Обрезаем нижние 20% изображения
            img = img.crop((0, 0, width, height - pixels_to_crop))
            # Сохраняем измененное изображение с тем же именем файла
            img.save(os.path.join(root, file))