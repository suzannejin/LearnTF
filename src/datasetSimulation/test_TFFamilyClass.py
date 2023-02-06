import unittest
import os
from TFFamilyClass import TfFamily

class TfFamilyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.TfFamily_object = TfFamily(pwm_file="test_TFFamilyData/test_PWM.txt", prot_file="test_TFFamilyData/test_prot_seq.txt")

    def test_parse_prot_shape(self) -> None:
        TF_prot_ID, prot = self.TfFamily_object._parseProt(self.TfFamily_object.prot_file)
        self.assertEqual(len(TF_prot_ID), len(prot), "ID // protein mismatch")

    def test_parse_prot(self) -> None:
        _, prot = self.TfFamily_object._parseProt(self.TfFamily_object.prot_file)
        self.assertEqual('MQKATYYDSSAIYGGYPYQAANGFAYNASQQPYAPSAALGTDGVEYHRPACSLQSPASAGGHPKTHELSEACLRTLSGPPSQPPGLGEPPLPPPPPQAAPPAPQPPQPPPQPPAPTPAAPPPPSSVSPPQSANSNPTPASTAKSPLLNSPTVGKQIFPWMKESRQNTKQKTSGSSSGESCAGDKSPPGQASSKRARTAYTSAQLVELEKEFHFNRYLCRPRRVEMANLLNLTERQIKIWFQNRRMKYKKDQKGKGMLTSSGGQSPSRSPVPPGAGGYLNSMHSLVNSVPYEPQSPPPFSKPPQGAYGLPPASYPAPLPSCAPPPPPQKRYTAAGSGAGGTPDYDPHAHGLQGNGSYGTPHLQGSPVFVGGSYVEPMSNSGPLFGLTHLPHTTSAAMDYGGTGPLGSGHHHGPGPGEPHPTYTDLTAHHPSQGRIQEAPKLTHL', prot[4], "incorrect protein match")
