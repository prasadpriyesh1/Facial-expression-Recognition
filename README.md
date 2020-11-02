# Facial-expression-Recognition
## Dataset
  I have used the following dataset from Kaggle</br>
  Dataset['https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset']</br>
  
  ** I was not able to upload the dataset because of the size constraint thus please download the data from the link and extract in same folder</br>
  
  Classes in Dataset
  * angry
  * disgust
  * fear
  * neutral
  * happy
  * sad
  * surprise
  
## Netwok used
  I have used the VGG16 network as my base network
## Transfer learning and fine tuning
  I have added 6 layers to the existing architecture</br>
  Layers Added
  * GlobalAveragePooling2D
  * Dense(2048 , activation ='relu')
  * Dense(1024 , activation ='relu')
  * Dense(512 , activation ='relu')
  * Dense(128 , activation ='relu')
  * Dense(7 , activation ='softmax')
## Input Shape
  * (224,224,3)
