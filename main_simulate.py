"""
main script for generating a simulated dataset
"""
import argparse
from src.datasetSimulation.simulate_data import SimulatedData
from src.datasetSimulation.TFFamilyClass import TfFamily

def get_args():
<<<<<<< HEAD
    "get the arguments when using from the commandline"
    
    return args
=======
    parser = argparse.ArgumentParser(description="Generate a simulation dataset")
    parser.add_argument("--ppm", type=str, help='path to the ppm file')
    parser.add_argument('--prot', type=str, help="path to the prot file")

    args = parser.parse_args()
    return args   
>>>>>>> main




if __name__ == "__main__":
    args = get_args()
<<<<<<< HEAD

=======
    tf_data = TfFamily(args.ppm, args.prot)
    simulated_data = SimulatedData(tf_data)
    simulated_data.simulate_data()
    
>>>>>>> main
    
