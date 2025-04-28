import flwr as fl
import os
import csv
import time

# Log file initiations
log_path = "fl_log_server.csv"
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Loss", "Accuracy", "Time(s)"])

# Custom strategy for Log
class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(fraction_fit=1.0, fraction_evaluate=1.0,
                         min_fit_clients=2, min_evaluate_clients=2,
                         min_available_clients=2)

    def aggregate_evaluate(self, server_round, results, failures):
        start_time = time.time()

        # get loss and metrics from EvaluateRes
        losses = []
        accuracies = []

        for client, eval_res in results:
            losses.append(eval_res.loss)
            if "accuracy" in eval_res.metrics:
                accuracies.append(eval_res.metrics["accuracy"])

        aggregated_loss = sum(losses) / len(losses) if losses else None
        aggregated_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        elapsed_time = time.time() - start_time
        print(f"Round {server_round} | Loss: {aggregated_loss} | Accuracy: {aggregated_accuracy} | Time: {elapsed_time}s")

        # Save to log file
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, aggregated_loss, aggregated_accuracy, round(elapsed_time, 2)])

        return aggregated_loss, {"accuracy": aggregated_accuracy}

def main():
    strategy = LoggingStrategy()

    fl.server.start_server(
        server_address="xxxxxx:xxxx", #fill your server address
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

