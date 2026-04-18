import socket
import pickle
import numpy as np
import onnxruntime as ort
import time
import statistics

PORT = 5005

# Loading the Predictor model, can replace the .onnx file with either the optimized or 
predictor = ort.InferenceSession("drone_predictor_64ch.onnx")

def summarize_metrics(inference_latencies):
    print("\n Ground Station Metrics Summary")

    # Setting up helper function to calculate the latency in ms
    def ms(x): return [v * 1000 for v in x]

    metrics = ("\n Ground Station Metrics Summary\n"
                f"{'Avg Inference (ms):':25} {statistics.mean(ms(inference_latencies)):.2f}"
              )


    print(metrics, flush=True)

    with open("ground_station_metrics.txt", "w") as f:
        f.write(avg)
        f.write("All done\n")

def start_ground_station():

    # Initializing variables for the metrics(total number of windows and a dictionary for storing the inference latency for each)
    total_windows = 0
    inference_latencies = []

    print("(Ground station) Waiting for features from drone")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            with conn:

                # Collecting the timestamp and features
                t_recv_start = time.time()

                data = []
                while True:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data.append(packet)

                # Loading the data from the tuple in the form of (drone_timestamp, features).
                t_inf_start = time.time()
                drone_timestamp, features = pickle.loads(b"".join(data))

    

                # Calculating the Predictor inference time for the given sensor window. 
                out = predictor.run(['predictor_logits'], {'feature_map_in': features})[0]
                prediction = np.argmax(out)
                t_inf_end = time.time()

                inference_latencies.append(t_inf_end - t_inf_start)

                total_windows += 1

                classes = ['Healthy', 'Motor Fault', 'Propeller Fault']
                print(f"\nGround Station: Received features from {addr}")
                print(f"Diagnosis: {classes[prediction]}")
                print(f"Inference Latency: {(t_inf_end - t_inf_start)*1000:.2f} ms")

    # Trying to get the results from the ground station on latency. 
    summarize_metrics(inference_latencies)


if __name__ == "__main__":
    start_ground_station()
    