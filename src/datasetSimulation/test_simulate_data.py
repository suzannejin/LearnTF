import unittest
from simulate_data import SimulatedData

class SimulateDataTestCase(unittest.TestCase):



    def test_onehote(sequence):
        sequence = "ACTG"
        encoding = [[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]]
        assert onehote(sequence) == encoding
        

    #TODO
    def test_score_dna_seq_with_ppm(self):
        self.assertEqual(1,1)
