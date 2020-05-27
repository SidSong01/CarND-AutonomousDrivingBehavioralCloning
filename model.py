  
import csv
import cv2
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dropout, Dense, Lambda, Conv2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils.vis_utils import plot_model


image_path = []
steering_angle = []
corr_parameter = 0.25     # correction value for Steering angle

# read the data
with open('./data/driving_log.csv') as csvfile:
    file = csv.reader(csvfile)


    fst_row = next(file)  
    for line in file:
        # Center camera
        image_path.append('./data/IMG/'+line[0].split('/')[-1])
        steering_angle.append(float(line[3]))
        
        # left camera
        image_path.append('./data/IMG/'+line[1].split('/')[-1])
        steering_angle.append(float(line[3])+corr_parameter)
        
        # Right camera
        image_path.append('./data/IMG/'+line[2].split('/')[-1])
        steering_angle.append(float(line[3])-corr_parameter)

# get the train and validation data
X_train_path, X_valid_path, y_train_angle, y_valid_angle = train_test_split(image_path,steering_angle,test_size=0.2)

# feed batch function for training process
def generator(X, y, batch_size):
    num_samples = len(X)
    while 1:
        X,y = shuffle(X,y)
        for offset in range(0, num_samples, int(batch_size/2)):
            batch_X, batch_y = X[offset:offset+int(batch_size/2)],y[offset:offset+int(batch_size/2)]
            images = []
            angles = []
            for i in range(len(batch_X)):
                img = cv2.imread(batch_X[i])
                images.append(img)
                angle = batch_y[i]
                angles.append(angle)
                # Mirror images and angles
                image_flipped = np.fliplr(img)
                images.append(image_flipped)
                angles.append(-angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
batch_size = 128
train_generator = generator(X_train_path,y_train_angle, batch_size=batch_size)
validation_generator = generator(X_valid_path,y_valid_angle, batch_size=batch_size)

# model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping = ((50, 20), (0,0))))
model.add(Conv2D(filters=24,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=36,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=48,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
plot_model(model, to_file="model_architecture.png",show_shapes=True);

# train the model
model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=np.ceil(2*len(X_train_path)/batch_size), 
            validation_data=validation_generator, 
           validation_steps=np.ceil(2*len(X_valid_path)/batch_size), 
            epochs=20, verbose=True)

# save the model
model.save('./model_2.h5')
