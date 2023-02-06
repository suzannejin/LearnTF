import unittest
from simulate_data import SimulatedData

class SimulateDataTestCase(unittest.TestCase): 
    

    def test_dna_seq_random_generator(self):
        SimulatedData.generate_list_dna_seq


    #TODO 
    def test_onehote():
        sequence = "ACTG"
        encoding = [[1,0,0,0], 
                    [0,1,0,0], 
                    [0,0,1,0],
                    [0,0,0,1]]
        assertEqual(onehote(sequence), encoding)
    
    #TODO
    def test_score_dna_seq_with_pwm(self):
        self.assertEqual(1,1)