import unittest
from data_TfFamilyClass import TfFamily
from data_SimulatedDataClass import SimulatedData
import numpy as np 

class SimulateDataTestCase(unittest.TestCase):
    def setUp(self):
            self.SData = SimulatedData(TfFamily("test_TFFamilyData/test_PWM.txt","test_TFFamilyData/test_prot_seq.txt" ))

    def test_onehote(self):
        sequence = "acgta"
        encoding = np.array([[1,0,0,0,1],
                    [0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0]])
        sequence = self.SData._sequence_to_array(sequence)
        encoded_sequence = self.SData._onehote(sequence)
        self.assertTrue(np.array_equal(encoded_sequence, encoding, equal_nan=True))
        

    def test_convolve(self):
        seq1 = "aa"
        seq2 = "tc"

        pwm = [[-5000,-1,0,2],
                [1,0,0,1]]
        pwm = np.array(pwm).T

        conv1 = self.SData._convolve(pwm,self.SData._onehote( self.SData._sequence_to_array(seq1)))[0]
        conv2 = self.SData._convolve(pwm,self.SData._onehote(self.SData._sequence_to_array(seq2)))[0]
        print(conv1)
        self.assertEqual(conv1,-4999)
        self.assertEqual(conv2,2)
        
