import flwr as fl
import utils
import pandas as pd
import keras
import tensorflow as tf

# df = pd.read_csv("train_split2.csv")
# # X_test, y_test = utils.preprocess(df)
# X_train, X_test, y_train, y_test = utils.preprocess(df)
# X_test = X_train + X_test
# y_test = y_train + y_test

from flwr.common import NDArrays, Scalar, EvaluateRes, FitRes

from typing import Dict, Optional, Tuple, Union, List
from flwr.server.client_proxy import ClientProxy

loaded_model = tf.keras.models.load_model("initialised_model.h5")
# Get model weights as a list of NumPy ndarray's
weights = loaded_model.get_weights()

# Serialize ndarrays to `Parameters`
parameters = fl.common.ndarrays_to_parameters(weights)

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
    
strategy = AggregateCustomMetricStrategy(
    # (same arguments as FedAvg here)
    min_available_clients=2,
    initial_parameters=parameters,
)
# Start Flower server for four rounds of federated learning
fl.server.start_server(server_address="localhost:3000", strategy=strategy, config=fl.server.ServerConfig(num_rounds=5))