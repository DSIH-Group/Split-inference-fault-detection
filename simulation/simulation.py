from mininet.net import Mininet
from mininet.node import Controller, OVSController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel
import time
import os

# Main function for running the simulation: 
def run_automated_simulation():
    # Clearing out any previous zombie processes and interfaces before running the simulation
    os.system('sudo mn -c > /dev/null 2>&1')
    
    # Setting up the virtual network switch
    net = Mininet(controller=OVSController, link=TCLink, switch=OVSSwitch)
    setLogLevel('info')
    
    # Using an existing virtual environment
    ENV_PYTHON = '/home/ashwins/drone/env/bin/python3'
    CUR_DIR = '/home/ashwins/drone/exportonnx2'
    
    print("Adding Hosts and virtual Switch")
    h1 = net.addHost('h1', ip='10.0.0.1') # Process representing drone with sentry and encoder running onboard. 
    h2 = net.addHost('h2', ip='10.0.0.2') # Process representing the ground_station containing the larger predictor model. 

    # Setting the switch to standlone mode to allow traffic even if controller is slow
    s1 = net.addSwitch('s1', failMode='standalone') 

    # Creating the simulated virtual wireless network between the 2 processes. 
    print("Created simulated wireless link between the 2 processes")
    net.addLink(h1, s1, delay='5ms', bw=54)
    net.addLink(h2, s1, delay='1ms', bw=1000)
    
    net.start()
    
    time.sleep(3)
    
    print("\nVerifying Connectivity...")
    # Using a manual ping to verify virtual connection. 
    ping_result = h1.cmd('ping -c 3 10.0.0.2')
    print(ping_result)
    
    # Running the simulation with both the ground station and drone processes
    try:
        print("\nLaunching Ground Station (h2):")
        ground_station = h2.popen(f'{ENV_PYTHON} ground_station.py', stdout=None, stderr=None)
        time.sleep(2)
        
        print("Starting Drone Simulation (h1):")
        
        # Checks if the drone process runs properly
        output = h1.cmd(f'cd {CUR_DIR} && {ENV_PYTHON} drone_edge.py')
        print(output)
        
    finally:
        print("\nSimulation Finished. Tearing down network.")
        net.stop()

if __name__ == '__main__':
    run_automated_simulation()