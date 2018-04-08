from keras.layers import Conv2D, Dropout, Conv2DTranspose, Cropping2D, ZeroPadding2D, Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
# from dataset import data

def VGG16():
    inputs = Input(shape=(480,320,3))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu',
               padding='valid', name='block1_conv1')(inputs)
    x = ZeroPadding2D(100)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    model = Model(inputs, x, name='vgg16')
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    return model
vgg16 = VGG16()
inputs = Input(shape=(480, 320, 3))
x = vgg16(inputs)
x = Conv2D(4096, kernel_size=7, activation='relu')(x)
x = Dropout(0.5)(x)
x = Conv2D(4906, kernel_size=1, activation='relu')(x)
x = Dropout(0.5)(x)
x = Conv2D(21, kernel_size=1)(x)
x = Conv2DTranspose(21, kernel_size=64, strides=32, use_bias=False,activation='softmax')(x)
x = Cropping2D(16)(x)
model = Model(inputs, x)
sgd = SGD(lr=1e-10, momentum=.99, decay=.0005)
model = multi_gpu_model(model,4) 
model.compile(sgd, loss='categorical_crossentropy')
# i = data('../data/sbdd/dataset','train',20)
checkpointer = ModelCheckpoint(filepath='./tmp/weights.hdf5')
model.fit_generator(i,i.step_per_epoch(),1000,callbacks=[checkpointer])
