# The implement of 《FedAux: An Efficient Framework for Hybrid Federated Learning》

(forked from [gaborvecsei/Federated-Learning-Mini-Framework](https://github.com/gaborvecsei/Federated-Learning-Mini-Framework))

This repo contains a Federated Learning (FL) setup with the Keras (Tensorflow) framework. The purpose is to have the codebase with which you can run FL experiments easily, for both IID and Non-IID data.
The two main components are: Server and Client. The Server contains the model description, distributes the data and coordinates the learning. And for all the clients it summarizes the results to update it's own (global) model. The Clients have different random chunks of data and the model description with the global model's weights. From this initialized status they can start the training on their own dataset for a few iterations. In a real world scenario the clients are edge devices and the training is running in parallel.

## abstract
As an enabler of sixth-generation communication technology (6G), Federated Learning (FL) triggers a paradigm shift from ''connected things'' to ''connected intelligence''. FL implements on-device learning, where massive end devices jointly and locally train a model without private data leakage. However, FL suffers from problems of low accuracy and convergence rate when no data is shared to the central server and the data distribution is non-IID. In recent years, attempts have been made on hybrid FL, where very small amounts of data (e.g., less than 1\%) is shared from the participants. With the opportunities brought by shared data, we notice that the server is capable of receiving the data in order to assist the FL process and mitigate the challenge of non-IID. Notably, existing hybrid FL only applies the model-level technologies belonging to the traditional FL and does not make full use of the characteristics of shared data to make targeted improvements. In this paper, we propose FedAux, a novel hybrid FL method at knowledge-level, which utilizes shared data to construct an auxiliary model and then transfer general knowledge to traditional aggregated model for enhancing the accuracy of global model and speeding up the convergence of global model. We also propose two specific knowledge transfer strategies named c-transfer and i-transfer. We conduct extensive analysis and evaluation of our methods against the well-known FL methods, FedAvg and Hybrid-FL protocol. The results indicate that FedAux shows higher accuracy (10.89\%) and faster convergence rate compared with other methods.

## Usage
1. train auxiliary model with the shared data
`python train_share_model_cifar.py -n test`

`python train_share_model_fmnist.py -n test`

2. the implement of FedAux and baseline
`python federated_learning_cifar_c_transfer.py -n test`

`python federated_learning_cifar_i_transfer.py -n test`

`python federated_learning_cifar_hybrid.py -n test` (Hybrid FL protocol)

`python federated_learning_fmnist_c_transfer.py -n test`

`python federated_learning_fmnist_i_transfer.py -n test`

`python federated_learning_fmnist_hybrid.py -n test` (Hybrid FL protocol)

3. some code should be fixed on FedAux/fed_learn/weight_summarizer.py to imply FedAvg
