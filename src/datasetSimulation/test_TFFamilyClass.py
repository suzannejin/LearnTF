import unittest
import os
from TFFamilyClass import TfFamily
import numpy as np

class TfFamilyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.TfFamily_object = TfFamily(ppm_file="test_TFFamilyData/test_PWM.txt", prot_file="test_TFFamilyData/test_prot_seq.txt")

    def test_parse_prot_shape(self) -> None:
        TF_prot_ID, prot = self.TfFamily_object._parseProt(self.TfFamily_object.prot_file)
        self.assertEqual(len(TF_prot_ID), len(prot), "ID // protein mismatch")

    def test_parse_prot(self) -> None:
        _, prot = self.TfFamily_object._parseProt(self.TfFamily_object.prot_file)
        self.assertEqual('MQKATYYDSSAIYGGYPYQAANGFAYNASQQPYAPSAALGTDGVEYHRPACSLQSPASAGGHPKTHELSEACLRTLSGPPSQPPGLGEPPLPPPPPQAAPPAPQPPQPPPQPPAPTPAAPPPPSSVSPPQSANSNPTPASTAKSPLLNSPTVGKQIFPWMKESRQNTKQKTSGSSSGESCAGDKSPPGQASSKRARTAYTSAQLVELEKEFHFNRYLCRPRRVEMANLLNLTERQIKIWFQNRRMKYKKDQKGKGMLTSSGGQSPSRSPVPPGAGGYLNSMHSLVNSVPYEPQSPPPFSKPPQGAYGLPPASYPAPLPSCAPPPPPQKRYTAAGSGAGGTPDYDPHAHGLQGNGSYGTPHLQGSPVFVGGSYVEPMSNSGPLFGLTHLPHTTSAAMDYGGTGPLGSGHHHGPGPGEPHPTYTDLTAHHPSQGRIQEAPKLTHL', prot[4], "incorrect protein match")

    def test_get_pwm_from_ppm(self) -> None:
        bg = np.array([0.25,0.25,0.25,0.25])
        ppm = [[0.25,0.25,0.25,0.25],
                [0,0,0,1]]
        ppm = np.array(ppm).T
        infval = -5000
        pwm = self.TfFamily_object._get_pwm_from_ppm(ppm,bg,infval)
        test_pwm = np.array([[0,0,0,0],
                            [-5000,-5000,-5000,2]]).T
        self.assertTrue(np.array_equal(pwm, test_pwm))
        self.assertEqual(ppm.shape,pwm.shape)

