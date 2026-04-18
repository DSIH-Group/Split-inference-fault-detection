## Split inference for real-time fault identification in drones ##

Below is a breakdown of the folders and files inside of this project. 

- training_script.py->a .py file used to train the machine learning architecture and generate .pt files to pass through the ONNX framework. 
- requirements_pi.txt->a .txt file containing all of the python packages installed on the Raspberry Pi 5 used in this project. 
- inference_test.py->a .py file for testing the inference capabilities of the architecture before and after optimizations using the ONNX runtime framework. 
- export_to_onnx.py->a .py file for converting the trained pytorch model into a series of onnx files to run on the Raspberry Pi 5.
- quantization_test.py->a .py file for generating graphs for different quantization experiments in an attempt to optimize inference speed. 

- dataset_splits->a folder containing three .txt files describing the specific testcases that go into the training, validation, and testing datasets.
  - The full HIL dataset can be found here: https://www.kaggle.com/datasets/xianglile/rflymad-hil
  - Link to the full RflyMAD dataset: https://rfly-openha.github.io/documents/4_resources/dataset.html#rflymad-a-dataset-for-multicopter-fault-detection-and-health-assessment

Inside of the simulation folder are the following files: 
- HIL_ds.py->a .py containing the class of the HIL dataset.
- drone_edge.py->a .py file for running a process representing a drone in the simulation
- ground_station.py->a .py file for running the process representing the ground station in the simulation.
- simulation.py->a .py file for orchestrating the entire simulation.

To run the simulation, first build and activate your virtual environment using the requirements_pi.txt file. 
Next, run the following command: sudo python simulation.py


