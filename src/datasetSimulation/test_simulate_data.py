import unittest
from TFFamilyClass import TfFamily
from simulate_data import SimulatedData
import numpy as np 
import src.utils.preProcessing as pre


class SimulateDataTestCase(unittest.TestCase):
    def setUp(self):
            self.SData = SimulatedData(TfFamily("test_TFFamilyData/test_PWM.txt","test_TFFamilyData/test_prot_seq.txt" ))

    def test_convolve(self):
        seq1 = "aa"
        seq2 = "tc"

        pwm = [[-5000,-1,0,2],
                [1,0,0,1]]
        pwm = np.array(pwm).T

        conv1 = self.SData._convolve(pwm,pre.encode_dna(seq1))[0]
        conv2 = self.SData._convolve(pwm,pre.encode_dna(seq2))[0]

        self.assertEqual(conv1,-4999)
        self.assertEqual(conv2,2)
        
