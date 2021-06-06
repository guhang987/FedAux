from keras import backend as K
from keras import optimizers, losses, models, layers, initializers,Sequential,Input
from keras.applications.vgg16 import VGG16


def create_model(input_shape: tuple, nb_classes: int, init_with_imagenet: bool = False, learning_rate: float = 0.001):
    weights = None
    if init_with_imagenet:
        weights = 'imagenet'
    
    model = VGG16(input_shape=input_shape,
                  classes=nb_classes,
                  weights=weights,
                  include_top=False)
    # "Shallow" VGG for Cifar10

    x = model.get_layer('block3_pool').output
    x = layers.Flatten(name='Flatten')(x)
    # init = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=660)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(.2)(x)
    x = layers.Dense(nb_classes)(x)
    x = layers.Softmax()(x)
    model = models.Model(model.input, x)

    loss = losses.categorical_crossentropy
    # optimizer = optimizers.SGD(lr=learning_rate, decay=0.99)
    #做共享模型时 不decay
    # optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.99)
    model.compile(optimizer, loss, metrics=["accuracy"])
    #最后统一加载参数
    # model.load_weights("./model/init_model")

    return model

def create_model_with_a_model(input_shape: tuple, nb_classes: int, init_with_imagenet: bool = False, learning_rate: float = 0.01):
    weights = None
    if init_with_imagenet:
        weights = 'imagenet'
    
    model = VGG16(input_shape=input_shape,
                  classes=nb_classes,
                  weights=weights,
                  include_top=False)
    # "Shallow" VGG for Cifar10

    x = model.get_layer('block3_pool').output
    x = layers.Flatten(name='n_1')(x)
    # init = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=660)
    x = layers.Dense(512, activation='relu', name="n_2")(x)
    x = layers.Dropout(.2)(x)
    x = layers.Dense(nb_classes, name="n_3")(x)
    x = layers.Softmax()(x)
    model = models.Model(model.input, x)

    loss = losses.categorical_crossentropy
    optimizer = optimizers.SGD(lr=learning_rate, decay=0.99)
    #做共享模型时 不decay
    # optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #最后统一加载参数
    
    #model.get_layer('block1_conv1').tranable = False
    model.compile(optimizer, loss, metrics=["accuracy"])
    # model.load_weights("./model/a_model",by_name=True)

    return model


def create_model_cnn(input_shape: tuple, nb_classes: int, init_with_imagenet: bool = False, learning_rate: float = 0.01):

    model = Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=2))
    # model.add(layers.Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
    # model.add(layers.MaxPool2D(pool_size=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(nb_classes, activation='softmax'))

    loss = losses.categorical_crossentropy
    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer, loss, metrics=["accuracy"])
    #最后统一加载参数
    # model.load_weights("./model/init_model")
    # model.summary()
    return model
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 28, 28, 32)        320       
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 28, 14, 16)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 28, 14, 64)        4160      
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 28, 7, 32)         0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 6272)              0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 256)               1605888   
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 256)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                2570      
# =================================================================
# Total params: 1,612,938
# Trainable params: 1,612,938
# Non-trainable params: 0
def set_model_weights(model: models.Model, weight_list):
    for i, symbolic_weights in enumerate(model.weights):
       
        weight_values = weight_list[i]
        K.set_value(symbolic_weights, weight_values)
