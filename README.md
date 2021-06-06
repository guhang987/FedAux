# The implement of 《FedAux:Hybrid Federated Learning with Auxiliary Model》

(forked from gaborvecsei/Federated-Learning-Mini-Framework)

This repo contains a Federated Learning (FL) setup with the Keras (Tensorflow) framework. The purpose is to have the codebase with which you can run FL experiments easily, for both IID and Non-IID data.
The two main components are: Server and Client. The Server contains the model description, distributes the data and coordinates the learning. And for all the clients it summarizes the results to update it's own (global) model. The Clients have different random chunks of data and the model description with the global model's weights. From this initialized status they can start the training on their own dataset for a few iterations. In a real world scenario the clients are edge devices and the training is running in parallel.

