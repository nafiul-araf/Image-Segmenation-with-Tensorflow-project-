from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K

def my_unet(img_height, img_width, img_channels, num_classes):
  input=Input(shape=(img_height, img_width, img_channels))
  
  #encode (Convoluion) phase
  c1=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(input)
  d1=Dropout(0.2)(c1)
  c1=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d1)
  p1=MaxPooling2D((2, 2))(c1)

  c2=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p1)
  d2=Dropout(0.2)(c2)
  c2=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d2)
  p2=MaxPooling2D((2, 2))(c2)

  c3=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p2)
  d3=Dropout(0.2)(c3)
  c3=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d3)
  p3=MaxPooling2D((2, 2))(c3)

  c4=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p3)
  d4=Dropout(0.2)(c4)
  c4=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d4)
  p4=MaxPooling2D((2, 2))(c4)
  
  #Latent dimension
  latent=Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p4)
  d5=Dropout(0.3)(latent)
  latent=Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d5)

  #decode (De-Convolution) phase
  u1=Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(latent)
  conc1=Add()([u1, c4])
  c6=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc1)
  d6=Dropout(0.2)(c6)
  c6=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d6)

  u2=Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c6)
  conc2=Add()([u2, c3])
  c7=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc2)
  d7=Dropout(0.2)(c7)
  c7=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d7)

  u3=Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c7)
  conc3=Add()([u3, c2])
  c8=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc3)
  d8=Dropout(0.2)(c8)
  c8=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d8)

  u4=Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c8)
  conc4=Add()([u4, c1])
  c9=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc4)
  d9=Dropout(0.2)(c9)
  c9=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d9)

  output=Conv2D(num_classes, (1, 1), activation='softmax')(c9)

  model=Model(inputs=[input], outputs=[output])
  model.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
  
  return model

def my_unet2(img_height, img_width, img_channels, num_classes):
  input=Input(shape=(img_height, img_width, img_channels))
  
  #encode (Convoluion) phase
  c1=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(input)
  d1=Dropout(0.2)(c1)
  c1=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d1)
  p1=MaxPooling2D((2, 2))(c1)

  c2=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p1)
  d2=Dropout(0.2)(c2)
  c2=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d2)
  p2=MaxPooling2D((2, 2))(c2)

  c3=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p2)
  d3=Dropout(0.2)(c3)
  c3=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d3)
  p3=MaxPooling2D((2, 2))(c3)

  c4=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p3)
  d4=Dropout(0.2)(c4)
  c4=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d4)
  p4=MaxPooling2D((2, 2))(c4)
  
  #Latent dimension
  latent=Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p4)
  d5=Dropout(0.3)(latent)
  latent=Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d5)

  #decode (De-Convolution) phase
  u1=Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(latent)
  conc1=Add()([u1, c4])
  c6=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc1)
  d6=Dropout(0.2)(c6)
  c6=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d6)

  u2=Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c6)
  conc2=Add()([u2, c3])
  c7=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc2)
  d7=Dropout(0.2)(c7)
  c7=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d7)

  u3=Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c7)
  conc3=Add()([u3, c2])
  c8=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc3)
  d8=Dropout(0.2)(c8)
  c8=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d8)

  u4=Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c8)
  conc4=Add()([u4, c1])
  c9=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc4)
  d9=Dropout(0.2)(c9)
  c9=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d9)

  output=Conv2D(num_classes, (1, 1), activation='softmax')(c9)

  model=Model(inputs=[input], outputs=[output])
  model.compile(loss='categorical_crossentropy', 
                optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True), 
                metrics=['accuracy'])
  
  return model


def my_unet3(img_height, img_width, img_channels, num_classes):
  input=Input(shape=(img_height, img_width, img_channels))
  
  #encode (Convoluion) phase
  c1=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(input)
  d1=Dropout(0.2)(c1)
  c1=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d1)
  p1=MaxPooling2D((2, 2))(c1)

  c2=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p1)
  d2=Dropout(0.2)(c2)
  c2=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d2)
  p2=MaxPooling2D((2, 2))(c2)

  c3=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p2)
  d3=Dropout(0.2)(c3)
  c3=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d3)
  p3=MaxPooling2D((2, 2))(c3)

  c4=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p3)
  d4=Dropout(0.2)(c4)
  c4=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d4)
  p4=MaxPooling2D((2, 2))(c4)

  c5=Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p4)
  d5=Dropout(0.2)(c5)
  c5=Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d5)
  p5=MaxPooling2D((2, 2))(c5)
  
  #Latent dimension
  latent=Conv2D(1024, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(p5)
  d6=Dropout(0.3)(latent)
  latent=Conv2D(1024, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d6)

  #decode (De-Convolution) phase
  u1=Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(latent)
  conc1=Add()([u1, c5])
  c6=Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc1)
  d7=Dropout(0.2)(c6)
  c6=Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d7)

  u2=Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
  conc2=Add()([u2, c4])
  c7=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc2)
  d8=Dropout(0.2)(c7)
  c7=Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d8)

  u3=Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
  conc3=Add()([u3, c3])
  c8=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc3)
  d9=Dropout(0.2)(c8)
  c8=Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d9)

  u4=Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
  conc4=Add()([u4, c2])
  c9=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc4)
  d10=Dropout(0.2)(c9)
  c9=Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d10)

  u5=Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c9)
  conc5=Add()([u5, c1])
  c10=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(conc5)
  d11=Dropout(0.2)(c10)
  c10=Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same', activation='relu')(d11)

  output=Conv2D(num_classes, (1, 1), activation='softmax')(c10)

  model=Model(inputs=[input], outputs=[output])
  model.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
  
  return model