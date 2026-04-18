import socket
import numpy as np
import onnxruntime as ort
import pickle
import time
from torch.utils.data import DataLoader
from HIL_ds import HILDataset

SERVER_IP = '10.0.0.2'
PORT = 5005

# Loading the Encoder and Sentry modesl
encoder = ort.InferenceSession("drone_encoder_64ch.onnx")
sentry = ort.InferenceSession("drone_sentry_head_64ch.onnx")

def run_drone_simulation(test_split_path):
    test_dataset = HILDataset(test_split_path, stride=64)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_windows = len(test_dataset)
    transmitted_count = 0

    print(f"Drone: Starting simulation on {total_windows} sensor windows")

    for window, label, folder_id, fault_class in test_dataloader:
        window_np = window.numpy()

        # Starting local inference timing, this would include the time it takes to receive the packet and send it over to the Predictor. 
        t_local_start = time.time()

        features = encoder.run(['feature_map'], {'input_telemetry': window_np})[0]
        sentry_out = sentry.run(['sentry_logits'], {'feature_map_in': features})[0]

        

        # Running the trigger logic of the sentry
        sentry_probs = np.exp(sentry_out) / np.exp(sentry_out).sum()
        predictor_class = int(np.argmax(sentry_probs))
        predictor_conf = float(np.max(sentry_probs))

        THRESHOLD = 0.725
        is_anomaly = int((predictor_class == 1) and (predictor_conf >= THRESHOLD))

        # If the Sentry identifies the current sensor window as anamolous, then we transmit it to the Predictor. 
        if is_anomaly:
            transmitted_count += 1
            try:
                t_send = time.time()
                payload = pickle.dumps((t_send, features))

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect((SERVER_IP, PORT))
                    s.sendall(payload)

            except Exception as e:
                print(f"Connection failed: {e}")

        t_local_end = time.time()
        local_latency = t_local_end - t_local_start

        # Writing them metrics to a file for calcualtion later
        with open("drone_metrics.log", "a") as f:
            f.write(f"{time.time()},{local_latency},{is_anomaly}\n")

    print(f"Simulation Completed. Drone transmitted {transmitted_count}/{total_windows} sensor windows.")


def summarize_drone_log(path="drone_metrics.log"):
    import pandas as pd

    df = pd.read_csv(path, header=None,
                     names=["timestamp", "local_latency", "is_anomaly"])

    total_windows = len(df)
    anomaly_windows = df["is_anomaly"].sum()

    avg_local_latency = df["local_latency"].mean()

    print("\n=== Drone Metrics Summary ===")
    print(f"{'Total Windows:':25} {total_windows}")
    print(f"{'Number of anomaly windows:':25} {anomaly_windows}")
    print(f"{'Avg Local Latency (ms):':25} {avg_local_latency * 1000:.3f}")


if __name__ == "__main__":
    run_drone_simulation("/home/ashwins/drone/exportonnx2/test_split.txt")
    summarize_drone_log()