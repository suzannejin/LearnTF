"""
main script for generating a simulated dataset
"""
import argparse
from src.datasetSimulation.simulate_data import SimulatedData
from src.datasetSimulation.TFFamilyClass import TfFamily

def get_args():
    
    "get the arguments when using from the commandline"

    parser = argparse.ArgumentParser(description="This script simulates data from a starting set of sequences and position-probabilities-matrices (ppm). This script is thought to work at Transcription Factor (TF) family level, for ex. Homeodomain. In praactical terms it is a sandbox for model testing, to achieve this:\n first it reads and parse two files (one for sequences one for ppm) matching the TF_id found in them; second it generates DNA sequences based on the ppm extracted.")
    parser.add_argument("-p", "--ppm", type=str, required=True, metavar="FILE", help='path to the ppm file, the structure format of this file is better defined in the RADME.md of src/datasetSimulation/')
    parser.add_argument("-s", '--prot', type=str, required=True, metavar="FILE", help="path to the prot file, as above ")
    # optional flag example
    # parser.add_argument("-o", "--opt", type=int, required=False,)

    args = parser.parse_args()
    return args   




if __name__ == "__main__":
    args = get_args()
    tf_data = TfFamily(args.ppm, args.prot)
    simulated_data = SimulatedData(tf_data)
    simulated_data.simulate_data()
