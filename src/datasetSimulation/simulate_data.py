import numpy as np
import random
import math
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class SimulatedData:
    """

    This class constructs a simulation dataset from raw data.

    TODO: GOAL IS TO HAVE THE FOLLOWING TABLE AT THE END : 
    # identifier (tf-ppm id;) # sequences # prot sequence # score #
    # T095100_1.02;M0931_1.02 # ATCGTA..  #   MVHHMYTC..  # 0.98 
    #            ---          #     --    #      --       #
    
    Then, we will have a "ready to go" training dataset
    
    TODO: Mathys 07/02/23 -> I have modified a few function that were slow/didn't output the correct results; 
    including: 
    * scoring function that became _convolve (because maybe in the future we will have multiple scoring functions). 
    Note (TODO) _convolve function is naïve and would be over 30x more performant if it would use numpy array broadcasting.

    * one hot enconding function; now uses sklearn which doesn't need to specify an alphabet -> useful for encoding both protein and dna sequences

    What is left TODO: is to use these functions in order to build the affordmentioned table; I have also added helper functions in TfFamily in order to get tf-ppm id and prot sequences


    """

    def __init__(self, TfFamily):
        self.TfFamily = TfFamily


    def simulate_data(self, l=100, n=100):
        """ Simulate DNA sequences and score them to each PPM for the entire TF family.

        Params
        ------
        l       (int)   : The wanted length of the DNA sequence
        n       (int)   : Number of DNA sequences to be generated per PPM

        Output
        ------
        scores  (numpy array of float) : List scores matching a DNA sequence with a PPM
        """

        # create empty numpy array
        self.dna_seqs = self.generate_list_dna_seq_for_all_ppms(l, n)
        scores = np.empty((len(self.TfFamily.get_ppms()), len(self.TfFamily.get_ppms()) * len(self.dna_seqs)), dtype = float)

        # compute score for each PPM and DNA sequence
        for i,ppm in enumerate(self.TfFamily.get_ppms()):
            for j,seq in enumerate(self.dna_seqs):
                scores[i,j] = self.score_sequence(seq, ppm)

        self.scores = scores

    def generate_list_dna_seq_for_all_ppms(self, l=100, n=100):
        """ For each PPM, generate the enriched random DNA sequences. 
      
        Params
        ------
        ppm     (list of numpy float arrays) : List of position probability matrix. rows = ACGT, cols = positions
        l       (int)   : The wanted length of the DNA sequence
        n       (int)   : Number of DNA sequences to be generated

        Output
        ------
        dna_seqs (list of str)   : List of DNA sequences
        """
        dna_seqs = []
        identifiers = []
        prot_sequence = []
        ppm_ids = self.TfFamily.get_identifiers()
        for i, ppm in self.TfFamily.get_ppms():
            for _ in range(n):
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
        dna_seq = ''.join(random.choice('CGTA') for _ in range(l))
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
            chunk += current_bp

        # modify DNA sequence
        start = random.choice(range(len(dna_seq)-npos))
        end   = start + npos
        enriched_seq = list(dna_seq)
        enriched_seq[start:end] = chunk
        enriched_seq = ''.join(enriched_seq)

        return enriched_seq

    @staticmethod
    def _sequence_to_array(sequence):
        """
        This function converts a string of nucleotides in an numpy array

        :param sequence:
        :return sequence_array:


        """
        sequence = sequence.lower()
        sequence = re.sub('[^acgt]', 'z', sequence)
        sequence_array = np.array(list(sequence))
        return sequence_array

    @staticmethod
    def _onehote(sequence_array):
        """
        One-hot-encodes a DNA sequence

        sequence (np.array): input DNA sequence in a np array format

        onehot_encoded_seq (matrix): one-hot-encoded sequence
        """

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(sequence_array)
        onehot_encoder = OneHotEncoder(sparse=False, dtype=int, categories=[range(5)])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded = np.delete(onehot_encoded, -1, 1)
        return onehot_encoded.transpose()

    # TODO: Protein encoder that takes as input a sequences array and returns an encoded protein

    @staticmethod
    def _convolve(ppm, one_hot_seq):
        """
        Computes the convolution operation between a one hot encoded sequence (arr_2) and a ppm (arr_1)
        """
        W = ppm.shape[1] # Window size
        L = one_hot_seq.shape[1]-W+1 # Find out about output size
        out = np.zeros(L) # Create output
        for i in range(L): 
            # For each sliding window, compute sum(ppm * seqlet) // convolution
            out[i] = np.multiply(ppm, one_hot_seq[:,i:i+W]).sum()
        return out
