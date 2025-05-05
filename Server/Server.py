import flwr as fl
import os
import csv
import time
from flwr.server.server import fit_clients

# Log file path
log_path = "#Change-to-your-own-path"
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Fit Time (s)", "Loss", "Accuracy"])

class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2
        )
        self._fit_time = 0
        
    def aggregate_fit(self, server_round, results, failures):
        fit_end_time = time.time()
        fit_duration = fit_end_time - self._fit_start_time
        self._fit_time = fit_duration
        return super().aggregate_fit(server_round, results, failures)
    
    def aggregate_evaluate(self, server_round, results, failures):
        losses = []
        accuracies = []
        for client, eval_res in results:
            losses.append(eval_res.loss)
            if "accuracy" in eval_res.metrics:
                accuracies.append(eval_res.metrics["accuracy"])

        loss_avg = sum(losses) / len(losses) if losses else None
        acc_avg = sum(accuracies) / len(accuracies) if accuracies else None
        # Logging to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, round(self._fit_time, 2), loss_avg, acc_avg])
        print(f"[Round {server_round}] Fit: {self._fit_time:.2f}s | Eval: {eval_time:.2f}s | Loss: {loss_avg:.4f} | Acc: {acc_avg:.4f}")
        return loss_avg, {"accuracy": acc_avg}

    def configure_fit(self, server_round, parameters, client_manager):
        self._fit_start_time = time.time()
        return super().configure_fit(server_round, parameters, client_manager)
    
def main():
    strategy = LoggingStrategy()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
