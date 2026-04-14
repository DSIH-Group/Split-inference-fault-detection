import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# Helper function to help with calcualting the total lost 
def calculate_tc_metrics(tc_map):
    if not tc_map: return 0.0, 0.0
    correct = 0; total_loss = 0
    for fid in tc_map:
        winning_vote = np.bincount(tc_map[fid]['preds']).argmax()
        if winning_vote == tc_map[fid]['f_class']: correct += 1
        total_loss += sum(tc_map[fid]['losses']) / len(tc_map[fid]['losses'])
    return 100 * (correct / len(tc_map)), total_loss / len(tc_map)

# Setting up the dataloaders for each of the train, validation and test datasets. 
train_dataset = HILDataset('train_split.txt')
val_dataset = HILDataset('val_split.txt', stride=64)
test_dataset = HILDataset('test_split.txt', stride=64)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=6,pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=6,pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=6,pin_memory=True)

# Both the sentry and predictor model will be using CrossEntropyLoss
criterion = nn.CrossEntropyLoss(reduction='none')


# Setting up the predictor model. 
model = Predictor(in_channels=train_dataset[0][0].shape[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}

# Phase 1: Training the predictor and encoder model on the training dataset
print(f"\n--- Starting Phase 1: Training the predictor and encoder model ---")
for epoch in range(10):
    model.train()
    training_map = {} # Map for storing information about the training loss and accuracy for each epoch. 

    for windows, labels, folder_ids, fault_classes in train_dataloader:
        windows = windows.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(windows)
        loss = criterion(out, labels)
        loss.mean().backward(); 
        optimizer.step()

        ls, preds = loss.detach().cpu().numpy(), out.argmax(1).cpu().numpy()
        for i, folder_id in enumerate(folder_ids):

            if folder_id not in training_map:
                training_map[folder_id] = {'preds': [], 'f_class': fault_classes[i].item(), 'losses': []}

            training_map[folder_id]['preds'].append(preds[i])
            training_map[folder_id]['losses'].append(ls[i])

    model.eval()
    validation_map = {} # Map for storing information about the validation loss and accuracy for each epoch. 

    with torch.no_grad():
        for windows, labels, folder_ids, fault_classes in val_dataloader:
            windows = windows.to(device)
            labels = labels.to(device)
            out = model(windows)
            loss = criterion(out, labels)
            ls, preds = loss.cpu().numpy(), out.argmax(1).cpu().numpy()

            for i, folder_id in enumerate(folder_ids):
                if folder_id not in validation_map:
                    validation_map[folder_id] = {'preds': [], 'f_class': fault_classes[i].item(), 'losses': []}

                validation_map[folder_id]['preds'].append(preds[i])
                validation_map[folder_id]['losses'].append(ls[i])

    training_acc, training_loss = calculate_tc_metrics(training_map)
    validation_acc, validation_loss = calculate_tc_metrics(validation_map)
    history['t_loss'].append(training_loss); history['v_loss'].append(validation_loss)
    history['t_acc'].append(training_acc); history['v_acc'].append(validation_acc)
    print(f"Epoch {epoch+1:02d} | Train Loss: {training_loss:.4f} | Train Acc: {training_acc:.2f}% | Val Loss: {validation_loss:.4f} | Val Acc: {validation_acc:.2f}%")


# Phase 2: Training the sentry model on the dataset
print(f"\n--- Starting Phase 2: Training the sentry model ---")

# Freezing the predictor and encoder model. 
for param in model.parameters():
    param.requires_grad = False
model.eval()

# Configuring and training the sentry model. 
sentry = Sentry()
sentry = sentry.to(device)
sentry_optimizer = optim.AdamW(sentry.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(5):
    sentry.train()
    for windows, labels, folder_ids, fault_classes in train_dataloader:
        windows = windows.to(device)
        labels = labels.to(device)
        sentry_optimizer.zero_grad()
        labels_binary = (labels > 0).long()

        with torch.no_grad():
            features = model.res_stack(model.stem(windows))

        out = sentry(features)
        loss = criterion(out, labels_binary)
        loss.mean().backward(); 
        sentry_optimizer.step()

    sentry.eval()
    sentry_validation_map = {}

    with torch.no_grad():
        for windows, labels, folder_ids, fault_classes in val_dataloader:
            windows = windows.to(device)
            labels = labels.to(device)
            labels_binary = (labels > 0).long()
            features = model.res_stack(model.stem(windows))
            out = sentry(features)
            loss = criterion(out, labels_binary)

            ls, preds = loss.cpu().numpy(), out.argmax(1).cpu().numpy()
            
            for i, folder_id in enumerate(folder_ids):    
                binary_faultclass = 1 if fault_classes[i].item() > 0 else 0
                if folder_id not in sentry_validation_map:
                    sentry_validation_map[folder_id] = {'preds': [], 'f_class': binary_faultclass, 'losses': []}
                
                sentry_validation_map[folder_id]['preds'].append(preds[i]) 
                sentry_validation_map[folder_id]['losses'].append(ls[i])

    sentry_acc, sentry_loss = calculate_tc_metrics(sentry_validation_map)
    print(f"Phase 2 Epoch {epoch+1:02d} | Val Loss: {sentry_loss:.4f} | Val TC Acc: {sentry_acc:.2f}%")


# Phase 3: Fine tuning the predictor model
print(f"\n--- Phase 3: Fine tuning the predictor model ---")

# Freezing the sentry and classifier models
for param in model.classifier.parameters():
    param.requires_grad = True

for param in sentry.parameters():
    param.requires_grad = False


# Setting up an optimizer with a low learning rate to prevent the predictor from forgetting the phase 1 training. 
phase3_optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)

# Fine tuning the predictor model
for epoch in range(5):
    model.classifier.train()
    model.stem.eval()
    model.res_stack.eval()

    for windows, labels, folder_ids, fault_classes in train_dataloader:
        windows = windows.to(device)
        labels = labels.to(device)
        phase3_optimizer.zero_grad()

        with torch.no_grad():
            features = model.res_stack(model.stem(windows))

        out = model.classifier(features)
        loss = criterion(out, labels)
        loss.mean().backward()
        phase3_optimizer.step()

    model.classifier.eval()
    predictor_validation_map_phase3 = {}

    with torch.no_grad():
        for windows, labels, folder_ids, fault_classes in val_dataloader:
            windows = windows.to(device)
            labels = labels.to(device)
            features = model.res_stack(model.stem(windows))
            out = model.classifier(features)
            loss = criterion(out, labels)

            preds, ls = out.argmax(1).cpu().numpy(), loss.cpu().numpy()
            for i, folder_id in enumerate(folder_ids):
                if folder_id not in predictor_validation_map_phase3:
                    predictor_validation_map_phase3[folder_id] = {'preds': [], 'f_class': fault_classes[i].item(), 'losses': []}
                
                predictor_validation_map_phase3[folder_id]['preds'].append(preds[i])
                predictor_validation_map_phase3[folder_id]['losses'].append(ls[i])

    phase3_acc, phase3_loss = calculate_tc_metrics(predictor_validation_map_phase3)
    print(f"Phase 3 Epoch {epoch+1:02d} | Val Loss: {phase3_loss:.4f} | Val TC Acc: {phase3_acc:.2f}%")


# Phase 4: Evaluating the entire architecture the test dataset
print(f"\n--- Phase 4: Evaluation of the architecture on unseen test data ({len(test_dataset.samples)} samples) ---")

model.eval()
sentry.eval()

# Maps for storing accuracy and loss of the predictor and encoder and sentry models respectively. 
test_predictor_map = {}
test_sentry_map = {}

with torch.no_grad():
    for windows, labels, folder_ids, fault_classes in test_dataloader:
        windows, labels = windows.to(device), labels.to(device)
        
        features = model.res_stack(model.stem(windows))
        predictor_out = model.classifier(features)
        predictor_loss = criterion(predictor_out, labels)
        predictor_preds = predictor_out.argmax(1).cpu().numpy()
        
        sentry_out = sentry(features)
        labels_binary = (labels > 0).long()
        sentry_loss = criterion(sentry_out, labels_binary)
        sentry_preds = sentry_out.argmax(1).cpu().numpy()

        for i, folder_id in enumerate(folder_ids):
            if folder_id not in test_predictor_map: 
                test_predictor_map[folder_id] = {'preds': [], 'f_class': fault_classes[i].item(), 'losses': []}

            test_predictor_map[folder_id]['preds'].append(predictor_preds[i])
            test_predictor_map[folder_id]['losses'].append(predictor_loss[i].cpu().numpy())
            
            binary_faultclass = 1 if fault_classes[i].item() > 0 else 0

            if folder_id not in test_sentry_map:
                test_sentry_map[folder_id] = {'preds': [], 'f_class': binary_faultclass, 'losses': []}

            test_sentry_map[folder_id]['preds'].append(sentry_preds[i])
            test_sentry_map[folder_id]['losses'].append(sentry_loss[i].cpu().numpy())

final_predictor_acc, final_predictor_loss = calculate_tc_metrics(test_predictor_map)
final_sentry_acc, final_sentry_loss = calculate_tc_metrics(test_sentry_map)

print(f"Final Test Specialist Acc: {final_predictor_acc:.2f}% | Loss: {final_predictor_loss:.4f}")
print(f"Final Test Sentry Acc:     {final_sentry_acc:.2f}% | Loss: {final_sentry_loss:.4f}")


# Plotting and evaluation 

# Phase 1 accuracy and loss curves between training and validation dataset
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history['t_loss'], label='Train'); ax1.plot(history['v_loss'], label='Val'); ax1.set_title('Phase 1 Loss'); ax1.legend()
ax2.plot(history['t_acc'], label='Train TC Acc'); ax2.plot(history['v_acc'], label='Val TC Acc'); ax2.set_title('Phase 1 Accuracy'); ax2.legend()
plt.savefig('phase1_training_curves.png')
print("Saved: phase1_training_curves.png")
plt.close()

# Generating Confusion matrices across all phases of training and the testing phase. 
fig, axes = plt.subplots(1, 5, figsize=(28, 6))

# Phase 1 confusion matrix
predictor_trues = [validation_map[f]['f_class'] for f in validation_map]
predictor_preds = [np.bincount(validation_map[f]['preds']).argmax() for f in validation_map]
ConfusionMatrixDisplay.from_predictions(predictor_trues, predictor_preds, display_labels=['NoFault', 'Motor', 'Prop'], cmap='Blues', ax=axes[0])
axes[0].set_title("Phase 1: Initial Predictor trainin")

# Phase 2 confusion matrix
sentry_trues = [sentry_validation_map[f]['f_class'] for f in sentry_validation_map]
sentry_preds = [np.bincount(sentry_validation_map[f]['preds']).argmax() for f in sentry_validation_map]
ConfusionMatrixDisplay.from_predictions(sentry_trues, sentry_preds, display_labels=['Normal', 'Anomaly'], cmap='Greens', ax=axes[1])
axes[1].set_title("Phase 2: Sentry training")

# Phase 3 confusion matrix
predictor3_trues = [predictor_validation_map_phase3[f]['f_class'] for f in predictor_validation_map_phase3]
predictor3_preds = [np.bincount(predictor_validation_map_phase3[f]['preds']).argmax() for f in predictor_validation_map_phase3]
ConfusionMatrixDisplay.from_predictions(predictor3_trues, predictor3_preds, display_labels=['NoFault', 'Motor', 'Prop'], cmap='Purples', ax=axes[2])
axes[2].set_title("Phase 3: Fine-Tuned Predictor")

# Phase 4 confusion matrix
test_sentry_trues = [test_sentry_map[f]['f_class'] for f in test_sentry_map]
test_sentry_preds = [np.bincount(test_sentry_map[f]['preds']).argmax() for f in test_sentry_map]
ConfusionMatrixDisplay.from_predictions(test_sentry_trues, test_sentry_preds, display_labels=['Normal', 'Anomaly'], cmap='Greens', ax=axes[3])
axes[3].set_title("Final Test: Sentry")

# Phase 5 confusion matrix
test_predictor_trues = [test_predictor_map[f]['f_class'] for f in test_predictor_map]
test_predictor_preds = [np.bincount(test_predictor_map[f]['preds']).argmax() for f in test_predictor_map]
ConfusionMatrixDisplay.from_predictions(test_predictor_trues, test_predictor_preds, display_labels=['NoFault', 'Motor', 'Prop'], cmap='Reds', ax=axes[4])
axes[4].set_title("Final Test: Predictor")



plt.tight_layout()
plt.savefig('confusion_matrices.png')
print("Saved: confusion_matrices.png")
plt.close()


# Saving the .pt files containing the weights for different parts of the architecture. 
print("\n Phase 5: Saving Component Weights")

encoder_state = {
    'stem': model.stem.state_dict(),
    'res_stack': model.res_stack.state_dict()
}

torch.save(encoder_state, 'encoder_weights.pt')
torch.save(sentry.state_dict(), 'sentry_weights.pt')
torch.save(model.classifier.state_dict(), 'specialist_weights.pt')

print("All model weights have been exported for deployment.")


    


