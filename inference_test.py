import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import onnxruntime as ort
import onnx
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
from onnx_opcounter import calculate_params
from HIL_ds import HILDataset

# Setting a constant for the sentry threshold
SENTRY_THRESHOLD = 0.725 

# Helper function for calculating the softmax of the output of the sentry. 
def softmax(x):
    """Numerically stable softmax for NumPy."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Helper for optimizing the passed in onnx file. 
def optimize_onnx_model(input_path, output_path):
    if not os.path.exists(input_path): 
        return False
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path
    ort.InferenceSession(input_path, sess_options, providers=['CPUExecutionProvider'])
    return True

# Helper function for evaluating the performance of the models.
def evaluate_models(encoder_path, sentry_path, predictor_path, test_loader, prefix):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4

    try:
        encoder_sess = ort.InferenceSession(encoder_path, sess_options, providers=['CPUExecutionProvider'])
        sentry_sess = ort.InferenceSession(sentry_path, sess_options, providers=['CPUExecutionProvider'])
        predictor_sess = ort.InferenceSession(predictor_path, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load {prefix} ONNX models: {e}")
        return None, None, None

    onnx_test_map = {}
    total_inference_time = 0.0
    total_windows = 0
    predictor_calls = 0

    for window, label, folder_id, fault_class in test_loader:
        window_np = window.numpy()
        
        for i in range(window_np.shape[0]):
            single_window = x_np[i:i+1]
            start_time = time.time()
            
            # 1. Using Encoder to perform feature extraction
            features = encoder_sess.run(['feature_map'], {'input_telemetry': single_window})[0]
            
            # 2. Passing outputs of encoder into sentry and calculating a probability. 
            sentry_logits = sentry_sess.run(['sentry_logits'], {'feature_map_in': features})[0]
            sentry_probs = softmax(sentry_logits)
            
            # 3. Retrieving the Sentry prediction and the confidence of that prediction. 
            sentry_pred = np.argmax(sentry_probs, axis=1)[0]
            sentry_conf = np.max(sentry_probs, axis=1)[0]

            # 4. Trigger logic for sending the result to the predictor model. 
            if sentry_pred == 1 and sentry_conf >= SENTRY_THRESHOLD:
                predictor_logits = predictor_sess.run(['predictor_logits'], {'feature_map_in': features})[0]
                final_prediction = np.argmax(predictor_logits, axis=1)[0]
                predictor_calls += 1
            else:
                final_prediction = 0
                
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            total_windows += 1

            if folder_id not in onnx_test_map:
                onnx_test_map[folder_id] = {'preds': [], 'f_class': fault_class[i].item()}
            onnx_test_map[folder_id]['preds'].append(final_prediction)

    avg_latency_ms = (total_inference_time / total_windows) * 1000
    trigger_rate = (predictor_calls / total_windows) * 100
    
    onnx_trues = [onnx_test_map[f]['f_class'] for f in onnx_test_map]
    onnx_preds = [np.bincount(onnx_test_map[f]['preds']).argmax() for f in onnx_test_map]
    accuracy = 100 * np.mean(np.array(onnx_trues) == np.array(onnx_preds))

    '''Using the matrices generating from the model inference, 
       we generate a confusion matrix showing the performance of the overall system.
       This part of the code was mainly for debugging purposes. 
    ''' 
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        onnx_trues, onnx_preds,
        display_labels=['NoFault', 'Motor', 'Prop'],
        cmap='Blues', ax=ax
    )
    ax.set_title(f"{prefix} (Trigger Rate: {trigger_rate:.1f}%)")
    plt.savefig(f'onnx_confusion_matrix_{prefix.lower()}.png')
    plt.close()

    return accuracy, avg_latency_ms, trigger_rate

# Helper function to manually calculate the number of FLOPS inside of a given model. 
def get_model_flops(model_path):
    model = onnx.load(model_path)
    total_flops = 0
    
    # Identifying and going through stored tensors in the model
    for init in model.graph.initializer:
        
        # Getting the dimension of the given tensor
        dims = list(init.dims)
        
        '''
            If the dimensions of the tensor is greater than or equal to 2, 
            it most likly is used to perform MACs(Multiply Accumulate Operation) during inference.
        ''' 
        if len(dims) >= 2:
            # Calculating the number of weights inside of the tensor. 
            weight_count = np.prod(dims)
            
            '''
            Each weight implies a Multiply-Accumulate (MAC).
            To calcualte the total number of FLOPS we use the following formula: 
            1 MAC = 2 FLOPs
            '''
            total_flops += (weight_count * 2)
            
    return total_flops

# Helper function for running benchmark tests on the models before and after ONNX runtime optimizations
def run_benchmark():
    test_dataset = HILDataset('test_split.txt', stride=64)

    if len(test_dataset) == 0: 
        return

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Dictionary of paths to the different optimized and unoptimized pnnx models. 
    models = {
        "base": ["drone_encoder_64ch.onnx", "drone_sentry_head_64ch.onnx", "drone_predictor_64ch.onnx"],
        "opt": ["drone_encoder_64ch_opt.onnx", "drone_sentry_head_64ch_opt.onnx", "drone_predictor_64ch_opt.onnx"]
    }


    # Optimizing the models
    print("Optimizing the models")
    for base, opt in zip(models["base"], models["opt"]):
        if not optimize_onnx_model(base, opt):
            print(f"Unable to find file {base}")
            return
    print("Optimizations completed\n")

    print("Running the architecture with the base models")
    base_acc, base_latency, base_trigger = evaluate_models(*models["base"], test_loader, "Base")
    
    print("Running the architecture with the optimized models")
    opt_acc, opt_latency, opt_trigger = evaluate_models(*models["opt"], test_loader, "Optimized")

    # Generating the table for displaying the metrics
    print("\n" + "="*70)
    print(f"{'Metric':<25} | {'Base':<18} | {'Optimized':<18}")
    print("-" * 70)
    print(f"{'Diagnostic Accuracy':<25} | {base_acc:>16.2f}% | {opt_acc:>16.2f}%")
    print(f"{'Avg Latency (ms)':<25} | {base_latency:>16.4f} | {opt_latency:>16.4f}")
    print(f"{'Predictor Trigger Rate':<25} | {base_trigger:>16.2f}% | {opt_trigger:>16.2f}%")
    print("-" * 70)
    print(f"{'Encoder Size (KB)':<25} | {os.path.getsize(models['base'][0])/1024:>16.1f} | {os.path.getsize(models['opt'][0])/1024:>16.1f}")
    print(f"{'Encoder FLOPS':<25} | {get_model_flops('drone_encoder_64ch.onnx'):>16.1f} | {get_model_flops('drone_encoder_64ch_opt.onnx'):>16.1f}")
    print(f"{'Sentry Size (KB)':<25} | {os.path.getsize(models['base'][1])/1024:>16.1f} | {os.path.getsize(models['opt'][1])/1024:>16.1f}")
    print(f"{'Sentry FLOPS':<25} | {get_model_flops('drone_sentry_head_64ch.onnx'):>16.1f} | {get_model_flops('drone_sentry_head_64ch_opt.onnx'):>16.1f}")
    print(f"{'Predictor Size (KB)':<25} | {os.path.getsize(models['base'][2])/1024:>16.1f} | {os.path.getsize(models['opt'][2])/1024:>16.1f}")
    print(f"{'Predictor FLOPS':<25} | {get_model_flops('drone_predictor_64ch.onnx'):>16.1f} | {get_model_flops('drone_predictor_64ch_opt.onnx'):>16.1f}")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()
