from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.models import Model, Sequential
from keras.callbacks import CSVLogger

# Data Augmentation
train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.1,
                zoom_range=0.1,
                rotation_range=10,
                height_shift_range=0.1,
                width_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                  './face_images/train',
                  target_size=(64,64),
                  batch_size=16,
                  class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                  './face_images/test',
                  target_size=(64,64),
                  batch_size=16,
                  class_mode='categorical')

# Model
input_tensor = Input(shape=(64,64, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(128))
top_model.add(Activation('relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(128))
top_model.add(Activation('relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(128))
top_model.add(Activation('relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2))
top_model.add(Activation('softmax'))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

for layer in model.layers[:15]:
  layer.trainable = False
    
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])

csv_logger = CSVLogger('test' + '.csv')
history = model.fit_generator(
              train_generator,
              steps_per_epoch=16,
              epochs=10,
              validation_data=validation_generator,
              validation_steps=16,
              callbacks=[csv_logger])

model.save('test_model' + '.h5')

