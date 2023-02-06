from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class SimulatedData: 
    
    def __init__(self, TfFamily):
        self.TfFamily = TfFamily

    def compute_scores(, l=100, n=1000):
        """ Simulate DNA sequences and score them to each PWM for the entire TF family.
    
        Params
        ------
        l       (int)   : The wanted length of the DNA sequence
        n       (int)   : Number of DNA sequences to be generated

        Output
        ------
        scores  (numpy array of float) : List scores matching a DNA sequence with a PWM
        """
        
        # generate list of dna sequences
        self.dna_seqs = generate_list_dna_seq()

        # create empty numpy array 
        scores = np.empty((len(tfs), len(self.dna_seqs)), dtype = float)
        
        # calculate score for each TF PWM and DNA sequence
        for i,tfid in enumerate(tfs):
            pwm = self.TfFamily.tfid2pwm(tfid)
            for j,seq in enumerate(self.dna_seqs):
                scores[i,j] = score_dna_seq_with_pwm(seq, pwm)
                
        self.scores = scores

    @staticmethod
    def _generate_list_dna_seq(l=100, n=1000):
        """ Generate a list of n random DNA sequences, with length l.
        
        Params
        ------
        l       (int)   : The wanted length of the DNA sequence
        n       (int)   : Number of DNA sequences to be generated
        
        Output
        ------
        dna_seqs (list of str)   : List of random DNA sequences
        """
        dna_seqs = []
        for i in range(n):
            dna_seqs.append( generate_dna_seq(l) )
        return dna_seqs

    @staticmethod
    def _generate_dna_seq(l=100):
        """ Generate a random DNA sequence, with length l.
        
        Params
        ------
        l       (int)   : The wanted length of the DNA sequence
        
        Output
        ------
        dna_seq (str)   : Generated random DNA sequence
        """
        import random

        dna_seq = ''.join(random.choice('CGTA') for _ in xrange(l))

        return dna_seq

    @staticmethod
    def onehote(sequence):
        """
        One-hot-encodes a DNA sequence

        sequence (string): input DNA sequence

        onehot_encoded_seq (matrix): one-hot-encoded sequence
        """
        
        seq_array = np.array(list(sequence)) 

        #integer encode the sequence
        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array) 
        
        #one hot the sequence
        onehot_encoder = OneHotEncoder(sparse=False)
        
        #reshape because that's what OneHotEncoder likes
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
        
        return onehot_encoded_seq

    def score_dna_seq_with_pwm(self, seq, pwm):
        """
        Scores a DNA sequence given a PWM

        seq (string) : DNA sequence 
        pwm (matrix) : a pwm in form of numpy matrix

        score (float): score from the PWM for the input sequence 
        """
        
        # Convert sequence to a one-hot-encode (ACTG)
        mask = onehote(seq)

        # Extract scores from the PWM for each position 
        masked_pwm = np.ma.masked_array(pwm, mask=mask)

        # Sum them up to obtain the final score for the sequence
        score = sum(masked_pwm)

        return score
    










   


     







