import numpy as np
import random
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class SimulatedData:

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
        ppms = self.TfFamily.get_ppms()
        self.dna_seqs = self.generate_list_dna_seq_for_all_ppms(ppms, l, n)
        scores = np.empty((len(ppms), len(ppms) * len(self.dna_seqs)), dtype = float)

        # compute score for each PPM and DNA sequence
        for i,ppm in enumerate(ppms):
            for j,seq in enumerate(self.dna_seqs):
                scores[i,j] = self.score_sequence(seq, ppm)

        self.scores = scores

    def generate_list_dna_seq_for_all_ppms(self, ppms, l=100, n=100):
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
        for ppm in ppms:
            for i in range(n):
                dna_seq = self._generate_random_dna_seq(l)
                dna_seq = self._enrich_dna_seq_with_ppm(dna_seq, ppm)
                dna_seqs.append(dna_seq)
        return dna_seqs

    def score_sequence(self, sequence, ppm):
        """
        Scores a DNA sequence of length N given a PPM of length M
        Scans the DNA sequence and estimates the score for each chunk with the PPM.
        Returns the maximum score.

        seq (string) : DNA sequence
        ppm (matrix) : a ppm in form of numpy matrix

        score (float): maximum score of a DNA chunk on sequence seq as evaluated
                       by ppm.
        """

        start_index = 0
        ppm_length = ppm.shape[1]
        # Set score to the mininum
        sequence_score = -math.inf
        # Compute the score of all possible DNA chunks
        # Store the maximum
        while start_index < len(sequence)-ppm_length+1:
            chunk = sequence[start_index: int(start_index + ppm_length)]
            score = self.score_chunk_seq_with_ppm(chunk, ppm)
            if(sequence_score < score):
                sequence_score = score
            start_index+=1

        return sequence_score
    

    def score_chunk_seq_with_ppm(self, seq, ppm):
        """
        Scores a DNA sequence of length N given a PPM with N positions

        seq (string) : DNA sequence
        ppm (matrix) : a ppm in form of numpy matrix

        score (float): score from the PPM for the input sequence
        """
        
        # Convert sequence to a one-hot-encode (ACTG)
        mask = self._onehote(seq)

        # Extract scores from the PPM for each position
        masked_ppm = np.ma.masked_array(ppm, mask=mask)

        # Sum them up to obtain the final score for the sequence
        # If we were working with PPMs
        # score = sum(masked_pwm)
        score = np.ndarray.prod(masked_ppm, where = mask.astype(bool)).item()

        return score

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
    def _onehote(sequence):
        """
        One-hot-encodes a DNA sequence

        sequence (string): input DNA sequence

        onehot_encoded_seq (matrix): one-hot-encoded sequence
        """

        alphabet = ['A', 'C', 'G', 'T']
        seq_array = np.array(list(sequence))

        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in sequence]
        onehot_encoded = list()

        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)

        one_hot_encoded_sequence = np.array(onehot_encoded).transpose()

        return one_hot_encoded_sequence
