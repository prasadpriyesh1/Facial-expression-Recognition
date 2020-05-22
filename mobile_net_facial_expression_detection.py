#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications import vgg16


# In[ ]:


model = vgg16.VGG16(weights='imagenet'  ,include_top = False, input_shape=(224,224,3))
for x in model.layers:
    x.trainable = False


# In[ ]:


from keras_preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'images/images/train',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        'images/images/validation',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')


# In[ ]:


from keras.models import Model
from keras.layers import Dense
from keras.models import Sequential


# In[ ]:


top_model = model.output
from keras.layers import  GlobalAveragePooling2D


# In[ ]:


#top_model = GlobalAveragePooling2D()(top_model)
#top_model = Dense(2048, activation='relu')(top_model)
top_model = Dense(1024, activation='relu')(top_model)
top_model = Dense(512, activation='relu')(top_model)
top_model = Dense(128, activation='relu')(top_model)
top_model = Dense(7, activation='softmax')(top_model)


# In[ ]:


newmodel  = Model(inputs = model.input , outputs = top_model )


# In[ ]:


newmodel.summary()


# In[ ]:


newmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


newmodel.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=12,
        validation_data=test_set,
        validation_steps=800)


# In[ ]:


newmodel.save("expressin_recog.h5")


# In[ ]:




