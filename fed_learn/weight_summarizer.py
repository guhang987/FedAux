from typing import List, Optional

import numpy as np
from . import models


class WeightSummarizer:
    def __init__(self):
        pass

    def process(self,
                client_weight_list: List[List[np.ndarray]],
                epoch: int,
                test_acc: float,
                global_weights: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        raise NotImplementedError()


class FedAvg(WeightSummarizer):
    def __init__(self, nu: float = 1.0):
        """
        Federated Averaging

        :param nu: Controls the summarized client join model fraction to the global model
        """

        super().__init__()
        self.nu = nu

    def process(self,
                client_weight_list: List[List[np.ndarray]],
                epoch: int,
                test_acc: float,
                global_weights_per_layer: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        nb_clients = len(client_weight_list)
        print("nb_clients=",nb_clients)
        weights_average = [np.zeros_like(w) for w in client_weight_list[0]]
        dataset = "fmnist"
        if dataset == "fmnist":
            a_model = models.create_model_cnn((28, 28, 1), 10)
            a_model.load_weights("./model/bias_f_model")
        elif dataset == "cifar":
            a_model = models.create_model((32, 32, 3), 10)
            a_model.load_weights("./model/a_model")


        for layer_index in range(len(weights_average)):
            w = weights_average[layer_index]
            if global_weights_per_layer is not None:
                global_weight_mtx = global_weights_per_layer[layer_index]
            else:
                global_weight_mtx = np.zeros_like(w)
            for client_weight_index in range(nb_clients):
                client_weight_mtx = client_weight_list[client_weight_index][layer_index]

                # TODO: this step should be done at client side (client should send the difference of the weights)
                client_weight_diff_mtx = client_weight_mtx - global_weight_mtx

                w += client_weight_diff_mtx

            # here we transfer knowledge from auxiliary model to aggregated model       

            if(layer_index < 4 and epoch <  20 ):          
                weights_average[layer_index] = a_model.get_weights()[layer_index]
            elif(layer_index < 3 and epoch >= 20 and epoch < 30):           
                weights_average[layer_index] = a_model.get_weights()[layer_index] 
            elif(layer_index < 2 and epoch >= 30 and epoch < 40): 
                weights_average[layer_index] = a_model.get_weights()[layer_index]
            elif(layer_index < 1 and epoch >= 40 and epoch < 50): 
                weights_average[layer_index] = a_model.get_weights()[layer_index] 
            else:
            
                weights_average[layer_index] = (self.nu * w / nb_clients) + global_weight_mtx
        return weights_average
