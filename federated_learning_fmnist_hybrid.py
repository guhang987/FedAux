import json
from pathlib import Path
import tensorflow as tf
import numpy as np
from keras import datasets

import fed_learn

args = fed_learn.get_args()

fed_learn.set_working_GPU(str(args.gpu))

experiment_folder_path = Path(__file__).resolve().parent / "experiments" / args.name
experiment = fed_learn.Experiment(experiment_folder_path, args.overwrite_experiment)
experiment.serialize_args(args)

tf_scalar_logger = experiment.create_scalar_logger()

client_train_params = {"epochs": args.client_epochs, "batch_size": args.batch_size}



def model_fn():
    return fed_learn.create_model_cnn((28, 28,1), 10, init_with_imagenet=False, learning_rate=args.learning_rate)


weight_summarizer = fed_learn.FedAvg()
server = fed_learn.Server(model_fn,
                          weight_summarizer,
                          args.clients,
                          args.fraction)

weight_path = args.weights_file
if weight_path is not None:
    server.load_model_weights(weight_path)

server.update_client_train_params(client_train_params)
server.create_clients_plus()

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)




data_handler = fed_learn.DataHandler(x_train, y_train, x_test, y_test, fed_learn.CifarProcessor(), args.debug)
data_handler.assign_data_to_clients_plus(server.clients, args.data_sampling_technique)
x_test, y_test = data_handler.preprocess(data_handler.x_test, data_handler.y_test)
#记录上一轮准确率，发给设备
test_acc = 0.1
for epoch in range(args.global_epochs):
    print("Global Epoch {0} is starting".format(epoch))
    server.init_for_new_epoch()
    selected_clients = server.select_clients()

    fed_learn.print_selected_clients(selected_clients)
    #加载全局模型并发给所有设备
    for client in selected_clients:
        print("Client {0} is starting the training".format(client.id))
        # if epoch == 0:
        #     server.load_model_weights("./model/a_model_3")
        server.send_model(client)
        hist = client.edge_train(server.get_client_train_param_dict())
        server.epoch_losses.append(hist.history["loss"][-1])

        server.receive_results(client)

    server.summarize_weights(epoch,test_acc)

    epoch_mean_loss = np.mean(server.epoch_losses)
    server.global_train_losses.append(epoch_mean_loss)
    tf_scalar_logger.log_scalar("train_loss/client_mean_loss", server.global_train_losses[-1], epoch)
    print("Loss (client mean): {0}".format(server.global_train_losses[-1]))

    global_test_results = server.test_global_model(x_test, y_test)
    print("--- Global test ---")
    test_loss = global_test_results["loss"]
    test_acc = global_test_results["acc"]
    print("{0}: {1}".format("Loss", test_loss))
    print("{0}: {1}".format("Accuracy", test_acc))
    tf_scalar_logger.log_scalar("test_loss/global_loss", test_loss, epoch)
    tf_scalar_logger.log_scalar("test_acc/global_acc", test_acc, epoch)

    with open(str(experiment.train_hist_path), 'w') as f:
        json.dump(server.global_test_metrics_dict, f)

    # TODO: save only when a condition is fulfilled (validation loss gets better, etc...)
    server.save_model_weights(experiment.global_weight_path)

    print("_" * 30)
    # if(epoch == 600):
    #     break
