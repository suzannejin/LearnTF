import unittest


# TODO: sequence_to_array test and encode protein test

def test_encode_dna(self):
        sequence = "acgta"
        encoding = np.array([[1,0,0,0,1],
                    [0,1,0,0,0],
                    [0,0,1,0,0],
                    [0,0,0,1,0]])
        sequence = self.sequence_to_array(sequence)
        encoded_sequence = self.encode_dna(sequence)
        self.assertTrue(np.array_equal(encoded_sequence, encoding, equal_nan=True))
        
        sequence_2 = "ac"
        encoding_2 = np.array([[1,0],
                    [0,1],
                    [0,0],
                    [0,0]])

        sequence_2 = self.sequence_to_array(sequence_2)
        encoded_sequence_2 = self.encode_dna(sequence_2)
        self.assertTrue(np.array_equal(encoded_sequence_2, encoding,_2 equal_nan=True))





