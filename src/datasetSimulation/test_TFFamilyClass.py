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





    def test_parse_PPM_uniqueID(self) -> None:
        list_right_tfID = ['T095100_1.02;M0931_1.02','T095193_1.02;M1009_1.02','T095193_1.02;M1010_1.02','T095300_1.02;M1072_1.02','T095300_1.02;M1073_1.02','T095191_1.02;M1007_1.02','T095191_1.02;M6026_1.02','T095191_1.02;M6027_1.02','T095112_1.02;M0941_1.02','T095144_1.02;M0966_1.02','T095243_1.02;M1049_1.02','T095076_1.02;M0909_1.02','T095103_1.02;M0933_1.02','T095103_1.02;M2298_1.02']
        ppm_array, _ = self.TfFamily_object._parsePPM(self.TfFamily_object.ppm_file)
        self.assertEqual(list_right_tfID, ppm_array, 'tf_id;motif_id id are not as they should')

    def test_parse_PPM_matrix(self) -> None:
        correct_matrices1 = np.array([[0.166699405763842, 0.0795346062089556, 0.0571793160359669, 0.696586671991235], [0.6584596739074, 0.0729594115502783, 0.164006043904652, 0.10457487063767], [0.860813734100549, 0.0145541786527099, 0.0441625941606885, 0.0804694930860531], [0.0574969226153707, 0.0580747762943826, 0.0374022710826174, 0.847026030007629], [0.146492860047455, 0.115235365165322, 0.184375915103163, 0.553895859684061], [0.427709577734801, 0.0714104636884069, 0.368133045159062, 0.132746913417731], [0.333757514973395, 0.210695660784104, 0.276003825538343, 0.179542998704157], [0.279203219679517, 0.233210132335507, 0.202743149129111, 0.284843498855865], [0.269584088744403, 0.207863239142908, 0.189972696527082, 0.332579975585607]])
        correct_matrices2 = np.array([[0.19032981084905, 0.2468444580687, 0.175696559601808, 0.387129171480441], [0.129841660563778, 0.195647723372541, 0.0566173634315743, 0.617893252632106], [0.444933004243443, 0.170361682087967, 0.0742860899014068, 0.310419223767183], [0.549744221654696, 0.186690677957144, 0.0653301156359427, 0.198234984752217], [0.103995623647377, 0.213652062956817, 0.0532154526786451, 0.62913686071716], [0.1369237974149, 0.190456638221744, 0.307792097732923, 0.364827466630433], [0.328462857049612, 0.171472647364429, 0.3124435344838, 0.187620961102159], [0.18240478247992, 0.360044738348107, 0.216205273571278, 0.241345205600695], [0.241926265546845, 0.298459125921828, 0.210320591317401, 0.249294017213926]])
        correct_matrices3 = np.array([[0.330648254698893,  0.184119677790562,  0.282316839278863,  0.202915228231682], [0.2270809359417, 0.20560030686613,  0.38626774069812,  0.18105101649405], [0.22784810126582,  0.26697353279632,  0.30763329497507,  0.19754507096279], [0.328346758726509,  0.0556194859992328,  0.209819716148829,  0.406214039125429], [0.178749520521669,  0.189489835059459,  0.564633678557728,  0.0671269658611428], [0.56501726121979,  0.10088224012275,  0.20598388952819,  0.12811660912927], [0.0598388952819328,  0.412351361718449,  0.427694668200999,  0.10011507479862], [0.0571538166474881,  0.0, 0.0, 0.942846183352512], [0.0, 0.0, 1.0, 0.0], [0.948216340621404,  0.0, 0.0191791331031841,  0.0326045262754121], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.027617951668585,  0.001150747986191,  0.924817798235519,  0.046413502109705], [0.100882240122749,  0.542385884158036,  0.291139240506328,  0.0655926352128875], [0.24741081703107,  0.19140774836977,  0.0467970847717681,  0.514384349827391]])
        _, ppm_list = self.TfFamily_object._parsePPM(self.TfFamily_object.ppm_file)
        self.assertTrue(np.array_equal(correct_matrices1, ppm_list[0], equal_nan=True))
        self.assertTrue(np.array_equal(correct_matrices2, ppm_list[2], equal_nan=True))
        self.assertTrue(np.array_equal(correct_matrices3, ppm_list[13], equal_nan=True))
        self.assertEqual(14, len(ppm_list), 'there are not the same number of matices')

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


