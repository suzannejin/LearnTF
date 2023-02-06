import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class SimulatedData: 
    
    def __init__(self, TfFamily):
        self.TfFamily = TfFamily

    def simulate_data(self, l=100, n=1000):
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
        self.dna_seqs = self._generate_list_dna_seq_enriched_with_ppm()

        # create empty numpy array 
        scores = np.empty((len(tfs), len(self.dna_seqs)), dtype = float)
        
        # calculate score for each TF PPM and DNA sequence
        for i,tfid in enumerate(tfs):
            ppm = self.TfFamily.tfid2ppm(tfid)
            for j,seq in enumerate(self.dna_seqs):
                scores[i,j] = score_dna_seq_with_ppm(seq, ppm)
                
        self.scores = scores

    @staticmethod
    def _generate_list_dna_seq_enriched_with_ppm(ppm, l=100, n=1000):
        """ Generate a list of n random DNA sequences, with length l, and enriched by a given PWM.
        
        Params
        ------
        ppm     (numpy float array) : Position probability matrix. rows = ACGT, cols = positions
        l       (int)   : The wanted length of the DNA sequence
        n       (int)   : Number of DNA sequences to be generated
        
        Output
        ------
        dna_seqs (list of str)   : List of DNA sequences
        """
        dna_seqs = []
        for i in range(n):
            dna_seq = self._generate_random_dna_seq(l)
            dna_seq = self._enrich_dna_seq_with_ppm(dna_seq, ppm)
            dna_seqs.append(dna_seq)
        return dna_seqs

    @staticmethod
    def _generate_random_dna_seq(l=100):
        """ Generate a random DNA sequence, with length l.
        
        Params
        ------
        l       (int)   : The wanted length of the DNA sequence
        
        Output
        ------
        dna_seq (str)   : Generated random DNA sequence
        """
        dna_seq = ''.join(random.choice('CGTA') for _ in xrange(l))
        return dna_seq

    @staticmethod
    def _enrich_dna_seq_with_ppm(dna_seq, ppm):
        """ Enrich the DNA sequence with PPM. The starting position of the enriched region is also randomly selected.

        Params
        ------
        dna_seq (str)               : A random DNA sequence
        ppm     (numpy float array) : Position probability matrix. rows = ACGT, cols = positions
        
        Output
        ------
        enriched_seq (str) : The DNA sequence enriched with PPM
        """

        # create DNA chunk matching the given PPM matrix
        bps   = ['A', 'C', 'G', 'T']
        npos  = ppm.shape[1]
        chunk = []
        for i in range(npos):
            pos_distr  = ppm[:,i]
            current_bp = random.choices(bps, pos_distr)
            chunk.append(current_bp)

        # modify DNA sequence
        start = random.choice(range(len(dna_seq)-npos))
        end   = start + npos
        enriched_seq = list(dna_seq)
        enriched_seq[start:end] = chunk
        enriched_seq = ''.join(enriched_seq)

        return enriched_seq

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

    @staticmethod
    def _score_dna_seq_with_pwm(seq, pwm):
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
    










   


     







