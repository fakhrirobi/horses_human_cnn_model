import tensorflow as tf
import numpy as np 
import urllib.request
import zipfile
import os

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"

file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(url,file_name)

if not os.path.exists(training_dir):
    os.makedirs(training_dir)

zip_ref = zipfile.ZipFile(file_name,'r')
zip_ref.extractall(training_dir)
zip_ref.close()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    training_dir,target_size=(300,300),
    class_mode='binary'
)

#the problem statement is to classify horses and human ( which can be concluded as binary classification )
#by this term the CNN Structurre should only be ended with single dense neuron with activation of sigmoid
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2), # yang jadi pertanyaan berapa layer yang dibutuhkan terus brp number of neurons
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')

    ]
)






#adding validation data for horses and human dataset 
validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
validation_filename = 'validation-horse-or-human.zip'
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(url,validation_filename)


if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

zip_ref = zipfile.ZipFile(validation_filename,'r')
zip_ref.extractall(validation_dir)
zip_ref.close()


validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,target_size=(300,300),
    class_mode='binary'

)
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])





history = model.fit_generator(train_generator,epochs=10,validation_data=validation_generator)




#saving models 
model.save('horse_human.h5')

