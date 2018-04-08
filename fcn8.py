from keras.layers import Conv2D, Dropout, Conv2DTranspose, Cropping2D, ZeroPadding2D, Input, MaxPooling2D, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
# from dataset import data


def fcn():
    inputs = Input(shape=(480, 480, 3))
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
    p3 = x
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    p4 = x
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # x = vgg16(inputs)
    score_p4 = Conv2D(21, 1,name='score_pool4')(p4)
    score_p3 = Conv2D(21, 1,name='score_pool3')(p3)
    x = Conv2D(4096, kernel_size=7, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, kernel_size=1, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(21, 1, name='score_fr')(x)
    x = Conv2DTranspose(21, 4, strides=2, padding='valid',
                        use_bias=False, name='upscore2')(x)

    score_p4c = Cropping2D(5)(score_p4)
    x = Add()([score_p4c, x])
    x = Conv2DTranspose(21, 4, strides=2, use_bias=False, name='upscore_pool4')(x)
    score_p3c = Cropping2D(9)(score_p3)
    x = Add()([score_p3c, x])
    x = Conv2DTranspose(21, kernel_size=16, strides=8,
                        use_bias=False, activation='softmax', name='upscore8')(x)
    x = Cropping2D(28)(x)
    model = Model(inputs, x)
    sgd = SGD(lr=1e-10, momentum=.99, decay=.0005)
    # model = multi_gpu_model(model, 4)
    model.compile(sgd, loss='categorical_crossentropy')
    return model
model = fcn()
# i = data('../data/sbdd/dataset','train',20)
# checkpointer = ModelCheckpoint(filepath='./tmp/weights.hdf5')
# model.fit_generator(i, i.step_per_epoch(), 1000, callbacks=[checkpointer])
