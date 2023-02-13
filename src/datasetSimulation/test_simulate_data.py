import unittest
from TFFamilyClass import TfFamily
from simulate_data import SimulatedData
import numpy as np 

class SimulateDataTestCase(unittest.TestCase):
    def setUp(self):
            self.SData = SimulatedData(TfFamily("test_TFFamilyData/test_PWM.txt","test_TFFamilyData/test_prot_seq.txt" ))

    def test_onehote(self):
        sequence = "ACTG"
        encoding = [[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]]
        sequence = self.SData._sequence_to_array(sequence)
        #self.assertTrue(np.array_equal(self.SData._onehote(sequence) ,  encoding, equal_nan=True))
        self.assertTrue(encoding ,  encoding, equal_nan=True)
        

    # TODO:
    def test_score_dna_seq_with_ppm(self):
        self.assertEqual(1,1)
