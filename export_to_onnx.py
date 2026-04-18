import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import onnx

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

# Architecture classes #

class Residual1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.batch1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.batch1(self.conv1(x)))
        out = self.batch2(self.conv2(out))
        return self.relu(out + residual)

class Predictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), 
            nn.ReLU()
        )

        # Creates a series of dialated convolutions to help with identifying patterns in the dataset. 
        # The purpose of using these dialated convolutions is to help view 
        self.res_stack = nn.Sequential(
            Residual1D(64, dilation=1),
            Residual1D(64, dilation=2),
            Residual1D(64, dilation=4)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Acts as the equivalent of global average pooling in CNNs for models that work with 1D data. 
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Adding dropout as a form of regularization
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_stack(x)
        return self.classifier(x)

class Sentry(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # Acts as the equivalent of global average pooling in CNNs for models that work with 1D data. 
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2), # Adding dropout as a form of regularization
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.classifier(x)

# Exporting the onnx file

def export_to_onnx(model, dummy_input, path, input_names, output_names):
    print(f"🚀 Exporting {path}...")
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            path,
            export_params=True,
            opset_version=18,          
            do_constant_folding=True, 
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={input_names[0]: {0: 'batch_size'}}
        )

def run_export():
    # Initialize models for 27-channel telemetry
    full_predictor = Predictor(in_channels=27)
    sentry = Sentry()

    # Preparing to load the weights from the .pt files. 
    device = torch.device('cpu')
    
    if os.path.exists('encoder_weights.pt'):
        enc_dict = torch.load('encoder_weights.pt', map_location=device)
        full_predictor.stem.load_state_dict(enc_dict['stem'])
        full_predictor.res_stack.load_state_dict(enc_dict['res_stack'])
        print("Encoder weights loaded")
    
    if os.path.exists('predictor_weights.pt'):
        full_predictor.classifier.load_state_dict(torch.load('predictor_weights.pt', map_location=device))
        print("Predicted weights loaded")

    if os.path.exists('sentry_weights.pt'):
        sentry.load_state_dict(torch.load('sentry_weights.pt', map_location=device))
        print("Sentry weights loaded")

    # 1. Exporting Encoder
    # Extracts features for both Sentry and Predictor
    dummy_input= torch.randn(1, 27, 64)
    encoder = nn.Sequential(full_predictor.stem, full_predictor.res_stack)
    export_to_onnx(encoder, dummy_input, "drone_encoder_64ch.onnx", 
                   ['input_telemetry'], ['feature_map'])

    # 2. Exporting Sentry model
    dummy_input = torch.randn(1, 64, 64)
    export_to_onnx(sentry, dummy_input, "drone_sentry_head_64ch.onnx", 
                   ['feature_map_in'], ['sentry_logits'])

    # 3. Exporting the Predictor model
    export_to_onnx(full_predictor.classifier, dummy_input, "drone_predictor_64ch.onnx", 
                   ['feature_map_in'], ['predictor_logits'])

if __name__ == "__main__":
    run_export()
    print("Export complete")