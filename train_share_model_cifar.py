import json
from pathlib import Path
import numpy as np
from keras import datasets
from keras import utils
import fed_learn
from keras.callbacks import *
import random
args = fed_learn.get_args()
NUMBER_OF_CLASS = 10

fed_learn.set_working_GPU(str(args.gpu))


experiment_folder_path = Path(__file__).resolve().parent / "experiments" / args.name
experiment = fed_learn.Experiment(experiment_folder_path, args.overwrite_experiment)

experiment.serialize_args(args)

tf_scalar_logger = experiment.create_scalar_logger()

client_train_params = {"epochs": args.client_epochs, "batch_size": args.batch_size}


def model_fn():
    return fed_learn.create_model((32, 32, 3), NUMBER_OF_CLASS, init_with_imagenet=True, learning_rate=args.learning_rate)

weight_summarizer = fed_learn.FedAvg()
server = fed_learn.Server(model_fn,
                          weight_summarizer,
                          args.clients,
                          args.fraction)


weight_path = args.weights_file
if weight_path is not None:
    server.load_model_weights(weight_path)


server.update_client_train_params(client_train_params)


server.create_clients()
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()


data_handler = fed_learn.DataHandler(x_train, y_train, x_test, y_test, fed_learn.CifarProcessor(), args.debug)
data_handler.assign_data_to_clients(server.clients, args.data_sampling_technique)
x_test, y_test = data_handler.preprocess(data_handler.x_test, data_handler.y_test)
y_train = y_train.astype(np.int32)



x_train = x_train.astype(np.float32)
x_train /= 255.0


x_share,y_share = fed_learn.get_shared_data(x_train,y_train,0.01)
y_share = utils.to_categorical(y_share, NUMBER_OF_CLASS)
print(len((y_share)))



classSize = int(len((y_share))/NUMBER_OF_CLASS)
trainSize = int(classSize * 0.8)
testSize = int(classSize * 0.2)


index1 = []
x = list(range(trainSize))
for i in range(NUMBER_OF_CLASS):
    y = [u+classSize*i for u in x]
    index1.extend(y)

index2 = []
x = list(range(trainSize,classSize))
for i in range(NUMBER_OF_CLASS):
    y = [u+classSize*i for u in x]
    index2.extend(y)
x_share_ = x_share[index1]
y_share_ = y_share[index1]
x_val = x_share[index2]
y_val = y_share[index2]


index = list(range(trainSize*NUMBER_OF_CLASS))
random.seed(660)
random.shuffle(index)
x_share__ = x_share_[index]
y_share__ = y_share_[index]
index = list(range(testSize*NUMBER_OF_CLASS))
random.seed(660)
random.shuffle(index)
x_val = x_val[index]
y_val = y_val[index]

model_weights = "./model/aux_model"

checkpoint = ModelCheckpoint(model_weights, monitor='val_loss', verbose=0, save_best_only=True, mode='min',
                             save_weights_only=True)
a_model = fed_learn.create_model((32, 32, 3), NUMBER_OF_CLASS, init_with_imagenet=False,learning_rate=0.01)


plateau        = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=1e-4, mode='min')
early_stopping = EarlyStopping(monitor="val_loss", patience=15)
y_train = utils.to_categorical(y_train, NUMBER_OF_CLASS)
print(a_model.summary())
a_model.fit(x_share, y_share,
            validation_data=(x_val,y_val),
            epochs=5000,
            callbacks=[plateau, early_stopping], 
            verbose=1,
            batch_size=32)


a_model.save_weights("./model/aux_model", overwrite=True)

results = a_model.evaluate(x_test, y_test, batch_size=50, verbose=3)

print(results)


