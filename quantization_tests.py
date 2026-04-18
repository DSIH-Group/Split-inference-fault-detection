import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, QConfig, HistogramObserver, default_per_channel_weight_observer
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Forcing Pytroch to use the ARM-optimized QNNPACK engine for edge simulation
torch.backends.quantized.engine = 'qnnpack'

# Dataset
class HILDataset(Dataset):
    def __init__(self, dataset_file, window_size=64, stride=32):
        self.samples = []
        with open(dataset_file, 'r') as f:
            paths = [line.strip() for line in f if line.strip() and 'TestInfo.csv' not in line]

        for path in paths:
            if not os.path.exists(path): continue
            path_upper = path.upper()

            # Reading through the path name to identify the fault type. 
            if 'HIL-NOFAULT' in path_upper: fault_class = 0
            elif 'MOTOR' in path_upper: fault_class = 1
            elif 'PROP' in path_upper: fault_class = 2
            else: continue

            test_info_path = os.path.join(path, 'TestInfo.csv')
            if not os.path.exists(test_info_path): continue

            # Identifying the injection time. 
            test_info = pd.read_csv(test_info_path, index_col=0, header=None).T
            fault_time_sec = float(test_info['Fault injection time'].iloc[0])

            sensor_file = self._find_file(path, 'sensor_combined_0.csv')
            actuator_file = self._find_file(path, 'actuator_outputs_0.csv')

            # If there is either a sensor or actuator .csv file that is missing, we skip the testcase entirly. 
            if not sensor_file or not actuator_file: continue


            # Sorting the actuator and sensor .csv files by timestamp
            actuator_dataframe = pd.read_csv(actuator_file).sort_values('timestamp')
            sensor_dataframe = pd.read_csv(sensor_file).sort_values('timestamp')

            '''
            Since the actuator and sensor .csv files were generating using sensors with different sample rates,
            We merge a given row in the sensor .csv file with the closest row from the actuator .csv file to 
            make a single row in merged_dataframe. 
            ''' 
            merged_dataframe = pd.merge_asof(sensor_dataframe, actuator_dataframe, on='timestamp', direction='nearest')

            # Dropping the timestamp column
            timestamps, data = merged_dataframe['timestamp'].values, merged_dataframe.drop(columns=['timestamp']).values
            
            # Finding the injection row 
            injection_row = np.searchsorted(timestamps, timestamps[0] + (fault_time_sec * 1e6))
            
            folder_id = os.path.basename(path.rstrip('/'))

            '''
            Now that we have the injection row, we break down the merged_dataframe into 64-row windows. 
            Each window is then assigned a label. If the window occurs before the injection fault, 
            it's given a label of 0(representing healthy). 
            
            Any windows at or after the injection row are labelled as the fault class. 
            On the otherhand, all windows before the injection row are labelled as healthy. 
            '''
            for start in range(0, len(merged_dataframe) - window_size, stride):
                end = start + window_size
                
                # Creating the label for a given window. 
                label = 0 if end < injection_row else fault_class

                # If the window we are using contains the injection row, 
                if start < injection_row < end: continue

                '''
                Normalizing all values in a given window column by subtracting the value by the column mean 
                and dividing the value by the column standard deviation.
                '''
                window = (data[start:end, :] - data[start:end, :].mean(axis=0)) / (data[start:end, :].std(axis=0) + 1e-6)
                
                # Transposing the window so that it can be processed by the encoder. Also storing additional meta data about the window. 
                self.samples.append((window.T, label, folder_id, fault_class))

    def _find_file(self, directory, suffix):
        log_dir = os.path.join(directory, 'Log')
        target = log_dir if os.path.exists(log_dir) else directory

        for file in os.listdir(target):
            if file.endswith(suffix): return os.path.join(target, file)
        return None

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, index):

        window, label, folder_id, fault_class = self.samples[index]

        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long), folder_id, torch.tensor(fault_class, dtype=torch.long)

# 2. Quantization-aware architecture
class QuantizedResidual1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        out = self.relu(self.batch1(self.conv1(x)))
        out = self.batch2(self.conv2(out))
        return self.relu(self.skip_add.add(out, residual))

class QuantizedPredictor(nn.Module):
    def __init__(self, in_channels=27):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.res_stack = nn.Sequential(
            QuantizedResidual1D(64, dilation=1),
            QuantizedResidual1D(64, dilation=2),
            QuantizedResidual1D(64, dilation=4)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.quant(x)
        features = self.res_stack(self.stem(x))
        out = self.classifier(features)
        return self.dequant(out), self.dequant(features)

    # Helper function used during static quantization to fuse consecutive layers without the encoder. 
    def fuse_model(self):

        fuse_modules(self.stem, ['0', '1', '2'], inplace=True)
        for m in self.res_stack:
            if isinstance(m, QuantizedResidual1D):
                fuse_modules(m, ['conv1', 'batch1', 'relu'], inplace=True)
                fuse_modules(m, ['conv2', 'batch2'], inplace=True)

class QuantizedSentry(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.dequant(self.classifier(self.quant(x)))

# Helper function(s) for generating visualizations and performance before and after quantization
def evaluate_quantization(model_fp32, sentry_fp32, test_dataloader):
    print("\n Evaluating quantization performance (FP32 vs Dynamic vs Static)...")
    device = torch.device('cpu')
    model_fp32.to(device).eval()
    sentry_fp32.to(device).eval()

    streaming_dataloader = DataLoader(test_dataloader.dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

    def measure_performance(predictor_net, sentry_net, name):
        torch.save(predictor_net.state_dict(), f'temp_spec_{name}.pt')
        torch.save(sentry_net.state_dict(), f'temp_sent_{name}.pt')
        size_mb = (os.path.getsize(f'temp_spec_{name}.pt') + os.path.getsize(f'temp_sent_{name}.pt')) / (1024 * 1024)
        
        correct, total = 0, 0
        latencies_predictor = []
        latencies_sentry = []

        with torch.no_grad():
            dummy_tensor = torch.randn(1, 27, 64).to(device)

            # Warming up the architecture on dummy tensor to prepare it for inference speed tests. 
            for _ in range(50):
                _out, _features = predictor_net(dummy_tensor)
                _ = sentry_net(_features)

            for window, label, folder_id, fault_class in streaming_dataloader:
                window = window.to(device)
                
                # 1. Timing the inference of the predictor on the current indow
                t0_predictor = time.perf_counter()
                out_predictor, features = predictor_net(window)
                t1_predictor = time.perf_counter()
                
                # 2. Timing the inference of the Sentry on the current window
                t0_sentry = time.perf_counter()
                out_sentry = sentry_net(features)
                t1_sentry = time.perf_counter()
                
                # Record individual latencies
                latencies_predictor.append((t1_predictor - t0_predictor) * 1000)
                latencies_sentry.append((t1_sentry - t0_sentry) * 1000)
                
                # Setting up the trigger logic
                prob = torch.softmax(out_sentry, dim=1)[:, 1]
                triggers = prob >= 0.5
                final_preds = torch.zeros_like(label)
                if triggers.any():
                    final_preds[triggers] = out_predictor.argmax(1)[triggers]
                
                correct += (final_preds == label.cpu()).sum().item()
                total += label.size(0)

        acc = (correct / total) * 100
        avg_predictor_ms = np.mean(latencies_predictor)
        avg_sentry_ms = np.mean(latencies_sentry)
        
        print(f"      Specialist Mean Latency: {avg_predictor_ms:.3f} ms")
        print(f"      Sentry Mean Latency:     {avg_sentry_ms:.3f} ms")
        print(f"      Total Combined Latency:  {(avg_predictor_ms + avg_sentry_ms):.3f} ms")
        
        if os.path.exists(f'temp_spec_{name}.pt'): os.remove(f'temp_spec_{name}.pt')
        if os.path.exists(f'temp_sent_{name}.pt'): os.remove(f'temp_sent_{name}.pt')
            
        return acc, size_mb, avg_predictor_ms, avg_sentry_ms

    # 1. Measuring the performance of the original unquantized architecture
    print("   Running FP32 Baseline...")
    fp32_results = measure_performance(model_fp32, sentry_fp32, 'fp32')

    # 2. Measuring the performance of the architecture with dynamic quantization
    print("   Running Dynamic INT8 (Linear only)...")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Quantizing only the linear layers of the architecture
        dyn_predictor = torch.ao.quantization.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)
        dyn_sentry = torch.ao.quantization.quantize_dynamic(sentry_fp32, {nn.Linear}, dtype=torch.qint8)

    dyn_results = measure_performance(dyn_predictor, dyn_sentry, 'dynamic')

    # 3. Measuring the performance of the architecture with static quantization
    print("   Running Static INT8 (Conv + Linear)...")

    static_predictor = QuantizedPredictor().cpu()    
    static_predictor.eval()
    static_predictor.load_state_dict(model_fp32.state_dict())

    static_sentry = QuantizedSentry().cpu()
    static_sentry.eval()
    static_sentry.load_state_dict(sentry_fp32.state_dict())
    
    
    # Fusing layers
    static_predictor.fuse_model()
    
    adv_qconfig = QConfig(
        activation=HistogramObserver.with_args(reduce_range=True),
        weight=default_per_channel_weight_observer
    )
    
    static_predictor.qconfig = adv_qconfig
    static_sentry.qconfig = adv_qconfig
    
    torch.ao.quantization.prepare(static_predictor, inplace=True)
    torch.ao.quantization.prepare(static_sentry, inplace=True)
    
    with torch.no_grad():
        for i, (x, _, _, _) in enumerate(test_dataloader):
            m_out, features = static_predictor(x)
            static_sentry(features)
            if i > 50: 
                break 
            
    torch.ao.quantization.convert(static_predictor, inplace=True)
    torch.ao.quantization.convert(static_sentry, inplace=True)
    static_results = measure_performance(static_predictor, static_sentry, 'static')

    # 4. Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['FP32', 'Dynamic', 'Static']
    data = [fp32_results, dyn_results, static_results]

    # Plot Accuracy and Memory
    for i, (ax, title, unit) in enumerate(zip([ax1, ax2], ['Accuracy', 'Memory'], ['%', 'MB'])):
        vals = [d[i] for d in data]
        ax.bar(labels, vals, color=['#4285F4', '#FBBC05', '#34A853'])
        ax.set_title(f'{title} ({unit})')
        for j, v in enumerate(vals):
            ax.text(j, v + (max(vals)*0.02), f"{v:.3f}", ha='center')

    # Plotting Latency using a Stacked Bar Chart
    vals_predictor_lat = [d[2] for d in data]
    vals_sentry_lat = [d[3] for d in data]
    
    ax3.bar(labels, vals_sentry_lat, color='#FBBC05', label='Sentry')
    ax3.bar(labels, vals_predictor_lat, bottom=vals_sentry_lat, color='#4285F4', label='Specialist (Predictor)')
    ax3.set_title('Latency Breakdown (ms/window)')
    ax3.legend()
    
    # Added total latency labels on the bar chart
    totals = [s + p for s, p in zip(vals_sentry_lat, vals_predictor_lat)]
    for j, total in enumerate(totals):
        ax3.text(j, total + (max(totals)*0.02), f"{total:.3f} ms", ha='center', fontweight='bold')

    plt.suptitle('Comparison of Deployment Quantization Strategies (Streaming Mode)', fontsize=14)
    plt.tight_layout()
    plt.savefig('quantization_comparison_final.png', dpi=300)
    print("Results saved as 'quantization_comparison_final.png'")



# Main function for orchestrating the benchmarking and visualizations. 
def run_full_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = HILDataset('test_split.txt', stride=64)

    if len(test_dataset) == 0: return
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
    
    predictor = QuantizedPredictor().to(device)
    sentry = QuantizedSentry().to(device)
    
    if os.path.exists('encoder_weights.pt'):
        encoder = torch.load('encoder_weights.pt', map_location=device)
        print(type(encoder))
        predictor.stem.load_state_dict(encoder['stem'])
        predictor.res_stack.load_state_dict(encoder['res_stack'])
        predictor.classifier.load_state_dict(torch.load('specialist_weights.pt', map_location=device))
        sentry.load_state_dict(torch.load('sentry_weights.pt', map_location=device))
        print("Sentry weights loaded.")

    predictor.eval(); 
    sentry.eval()

    print("\n Generating Confidence threshold curve diagram")
    all_sentry_probs, all_sentry_preds, all_labels = [], [], []

    # Running the full architecture on the test dataset. We store the sentry confidence and hte 
    with torch.no_grad():
        for window, label, folder_id, fault_class in test_dataloader:
            window, label = window.to(device), label.to(device)
            out_predictor, features = predictor(window)
            sentry_prob = torch.softmax(sentry(features), dim=1)[:, 1]
            all_sentry_probs.extend(sentry_prob.cpu().numpy())
            all_sentry_preds.extend(out_predictor.argmax(1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Calculating the overall architecture accuracy and predictor model activation releative to the given sentry confidence threshold. 
    ts = np.linspace(0, 1, 1000)
    accs, rates = [], []
    for t in ts:
        trig = np.array(all_sentry_probs) >= t
        preds = np.zeros_like(all_labels)
        preds[trig] = np.array(all_sentry_preds)[trig]
        accs.append(np.mean(preds == np.array(all_labels)) * 100)
        rates.append(np.mean(trig) * 100)

    # Generating the table using the data calculated in the above loop. 
    plt.figure(figsize=(8, 5))

    # Represents the accuracy of the overall system
    plt.plot(ts, accs, label='Accuracy')

    # The percentage of windows that are sent to the predictor model
    plt.plot(ts, rates, label='Activation %')
    plt.title('Confidence threshold efficiency')
    plt.legend()
    plt.savefig('efficiency_curve.png')
    
    accs = np.array(accs, dtype=float)
    rates = np.array(rates, dtype=float)
    ts = np.array(ts, dtype=float)


    max_acc = accs.max()
    tolerance = 0.5  # allow 0.5% drop from max accuracy

    # Indices where accuracy is within tolerance of the max
    good_idxs = np.where(accs >= max_acc - tolerance)[0]

    # Among those, pick the one with minimum activation %
    best_idx = good_idxs[np.argmin(rates[good_idxs])]

    best_threshold = ts[best_idx]
    best_acc = accs[best_idx]
    best_rate = rates[best_idx]

    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Accuracy at best threshold: {best_acc:.2f}%")
    print(f"Activation at best threshold: {best_rate:.2f}%")


    # Performing quantization tests and generating the stacked bar charts for overall architecture accuracy, memory and inference speed. 
    evaluate_quantization(predictor, sentry, test_dataloader)

if __name__ == "__main__":
    run_full_analysis()