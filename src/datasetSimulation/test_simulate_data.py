import unittest
from TFFamilyClass import TfFamily
from simulate_data import SimulatedData

class SimulateDataTestCase(unittest.TestCase):
    def setUp(self):
            self.SData = SimulatedData(TfFamily("test_TFFamilyData/test_PWM.txt","test_TFFamilyData/test_prot_seq.txt" ))

    def test_onehote(self):
        sequence = "ACTG"
        encoding = [[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]]
        self.assertEqual(self.SData._onehote(sequence) , encoding)

        

    #TODO
    def test_score_dna_seq_with_ppm(self):
        self.assertEqual(1,1)
