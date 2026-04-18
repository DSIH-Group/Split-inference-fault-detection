import onnxruntime as ort
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np

class HILDataset(Dataset):
    def __init__(self, manifest_file, window_size=64, stride=32):
        self.samples = []
        if not os.path.exists(manifest_file):
            print(f"ERROR: {manifest_file} not found.")
            return

        with open(manifest_file, 'r') as f:
            base_paths = []
            for line in f:
                path = line.strip().split("] ")[-1] if "] " in line else line.strip()
                if path and not path.endswith('.csv'):
                    base_paths.append(path)

        for path in base_paths:
            if not os.path.exists(path):
                print(f"Missing path: {path}")
                continue
            
            # Identifying the fault class by accessing TestInfo.csv file. 
            info_file = os.path.join(path, 'TestInfo.csv')
            if not os.path.exists(info_file): continue
            
            info_dataframe = pd.read_csv(info_file, header=None)
            
            try:
                # Extracting the fault_id and the injection time. 
                fault_id = info_dataframe[info_dataframe[0] == 'Fault ID'][1].values[0]
                fault_time = float(info_dataframe[info_dataframe[0] == 'Fault injection time'][1].values[0])
            except (IndexError, ValueError):
                print(f"Metadata error inside of {info_file}")
                continue

            # Mapping the ids to the different fault classes (0: Healthy, 1: Motor, 2: Propeller)
            if '123450' in fault_id:
                fault_class = 1  # Motor
            elif '123451' in fault_id:
                fault_class = 2  # Propeller
            else:
                fault_class = 0  # NoFault

            sensor_file = self._find_file(path, 'sensor_combined_0.csv')
            actuator_file = self._find_file(path, 'actuator_outputs_0.csv')
            if not sensor_file or not actuator_file: continue

            sensor_dataframe = pd.read_csv(sensor_file).sort_values('timestamp')
            actuator_dataframe = pd.read_csv(actuator_file).sort_values('timestamp')
            merged_dataframe = pd.merge_asof(sensor_dataframe, actuator_dataframe, on='timestamp', direction='nearest')

            timestamps = merged_dataframe['timestamp'].values
            data = merged_dataframe.drop(columns=['timestamp']).values
            
            # Aligning the rows from both .csv files using their timestamps. 
            injection_row = np.searchsorted(timestamps, timestamps[0] + (fault_time * 1e6))
            folder_id = os.path.basename(path.rstrip('/'))

            for start in range(0, len(merged_dataframe) - window_size, stride):
                end = start + window_size
                label = 0 if end < injection_row else fault_class
                if start < injection_row < end: 
                    continue 
                
                window = (data[start:end, :] - data[start:end, :].mean(axis=0)) / (data[start:end, :].std(axis=0) + 1e-6)
                self.samples.append((window.T, label, folder_id, fault_class))

    # Helper function for finding a specific Log file based on a given suffic. 
    def _find_file(self, directory, suffix):
        search_paths = [os.path.join(directory, 'Log'), directory]
        for target in search_paths:
            if os.path.exists(target):
                for f in os.listdir(target):
                    if f.endswith(suffix): return os.path.join(target, f)
        return None

    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        window, label, folder_id, fault_class = self.samples[index]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long), folder_id, torch.tensor(fault_class, dtype=torch.long)
