# Imports
import keras
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential


from keras import applications
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing.image import load_img, img_to_array


from matplotlib import pyplot as plt

#%% 
#
# We first declare an ImageDataGenerator. This is an object that defines how images should be converted into NumPy data.
# In our case, we add a parameter rescale; this means that the color values range from 0 to 1 instead of 0 to 255. 

data_gen = ImageDataGenerator(rescale=1.0/255, rotation_range=0, shear_range=0,zoom_range=0 )


#%% 
#
# train the images (parameters below)
#the location of the training images;
#target_size is for the size of the images: in our case, we'll resize them so that they are all 64x64 pixels;
#batch_size refers to the batch size we'll use when training;
#class_mode='binary' means that we'll treat the learning problem as a binary classification problem;
#classes is provided to make sure that other is coded as 0 and car as 1;
#shuffle because we'd like the images to appear in a random order when training.

imgdir = 'a3_images'
img_size = 64
batch_size = 32

train_generator = data_gen.flow_from_directory(
        imgdir + '/train',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        shuffle=True)


validation_generator = data_gen.flow_from_directory(
        imgdir + '/validation',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=['other', 'car'],
        shuffle=True)


#%% 
#
# take a look at images
#

Xbatch, Ybatch = train_generator.next()

Xbatch_validation, Ybatch_validation = validation_generator.next()

print("shape of the batch is:\n", Xbatch.shape)



#%% 
#
# Plot
#

plt.imshow(Xbatch[5])




#%% 
#
# Training a convolutional neural network
#
#input shape should be (img_size, img_size, 3)


def make_convnet():
    num_classes = 1
    img_height = 64
    img_width = 64

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    model.fit(Xbatch, Ybatch,
              batch_size=32,
              epochs=1,
              verbose=1,
              validation_data=(Xbatch_validation, Ybatch_validation))


    score = model.evaluate(Xbatch_validation, Ybatch_validation, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    return model;



My_CNN = make_convnet();


fitted_gen = My_CNN.fit_generator(train_generator, validation_data=validation_generator, epochs=20)



#Optionally, call your_cnn.save_weights(some_file_name) after training. 
#This will save your weights to a file; you can recover them later using 
#your_cnn.load_weights(the_same_file_name), so that you can run your CNN 
#several times without having to re-train every time.
#%%

My_CNN.save_weights('My_weights')


#%% 
My_CNN.load_weights('My_weights')


#%%
#
# Plot the training and validation loss for each epoch. Also plot the training and validation accuracies in another plot.
#

#fitted_gen.evaluate_generator();

#print(fitted_gen.history)
#print(fitted_gen.history['val_loss'])

plt.figure("Loss");
plt.plot(fitted_gen.history['val_loss'], label="Validation_loss", color = 'K')
plt.plot(fitted_gen.history['loss'], label="Training_loss", color = 'Y')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)

plt.figure("Accuracy");
plt.plot(fitted_gen.history['val_acc'],label="Validation_Accuracy", color = 'R')
plt.plot(fitted_gen.history['acc'],label="Training_Accuracy", color = 'G')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)

        

#%%
#
# Uppgift 3
#


vggmodel = applications.VGG16(weights='imagenet', include_top=True)


#%%
#plt.imshow(Xbatch[5])

My_image = load_img('F:/Machine learning/a3_images/train/other/0034.jpg', target_size=(224,224))

plt.figure("Cow");
plt.imshow(My_image)

My_image_array = img_to_array(My_image)

Processed_image = preprocess_input(My_image_array)

My_image = Processed_image.reshape(1, 224, 224, 3)

guess = vggmodel.predict(My_image)

decodad_guess = decode_predictions(guess)

print("Prediction is :" , decodad_guess[0][0], " \n It is in reality the rear of a cow")


#%%
# Part 3: Using VGG-16 as a feature extractor ,

img_size = 64

feature_extractor = applications.VGG16(include_top=False, weights='imagenet',
                                       input_shape=(img_size, img_size, 3))


vgg_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

#%%

def  create_vgg16_features():
    imgdir = 'a3_images'
    img_size = 64
    batch_size = 32
    
    train_generator = vgg_data_gen.flow_from_directory(
            imgdir + '/train',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            classes=['other', 'car'],
            shuffle=False)
    
    
    validation_generator = vgg_data_gen.flow_from_directory(
            imgdir + '/validation',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            classes=['other', 'car'],
            shuffle=False)
      
    Xbatch, Ybatch = train_generator.next()
    
    Xbatch_validation, Ybatch_validation = validation_generator.next()
    
    
    return train_generator, validation_generator, Xbatch, Ybatch;



train_generator, validation_generator, Xbatch, Ybatch = create_vgg16_features();
    
    
#%%

fourD_array_train = feature_extractor.predict_generator(train_generator)

with open('My_train_file', 'wb') as f:
  np.save(f, fourD_array_train)
  
  
fourD_array_val = feature_extractor.predict_generator(validation_generator)

with open('My_vali_file', 'wb') as f:
  np.save(f, fourD_array_val)


#%%
  
  
def train_on_cnnfeatures():
     
    with open('My_train_file', 'rb') as f:
        the_train_data = np.load(f)

    with open('My_vali_file', 'rb') as f:
        the_vali_data = np.load(f)
        
    #trains a classifier
    
    #evaluates on the validation set
        

the_train_data, the_vali_data = train_on_cnnfeatures();

#%%

