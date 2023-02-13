import numpy as np
import pandas as pd
import random
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class SimulatedData:
    """

    This class constructs a simulation dataset from raw data.

    
    TODO: Mathys 07/02/23 -> I have modified a few function that were slow/didn't output the correct results; 
    including: 
    * scoring function that became _convolve (because maybe in the future we will have multiple scoring functions). 
    Note: TODO _convolve function is naïve and would much more performant if it was able to score EVERYTHING AT ONCE using np broadcasting

    * one hot enconding function; now uses sklearn which doesn't need to specify an alphabet -> useful for encoding both protein and dna sequences

    TODO: it seems that ppm augmented sequences do not produce a higher score than non-augmented ppm sequences when scanning for the associated ppm.


    """

    def __init__(self, TfFamily, l=100, n=100):
        """
        For each PPM, n DNA sequences are generated randomly and enriched by the given PPM motif.
        Thus, a total of n x number of PPMs.
        Then, each DNA sequence is scored to each PWM.

        TfFamily (class object)
        l        (int)   : The wanted length of the DNA sequence
        n        (int)   : Number of DNA sequences to be generated per PPM
        """
        self.TfFamily = TfFamily
        self.l = l
        self.n = n
        self.data = self.simulate_data()


    def simulate_data(self):
        """ Simulate DNA sequences and score them to each PPM for the entire TF family.

        Output
        ------
        scores  (numpy array of float) : List scores matching a DNA sequence with a PPM

        """
        
        # initialize data dictionary
        d = {
            'id'       : [],
            'prot_seq' : [],
            'ppm'      : [],
            'dna_seq'  : [],
            'label'    : [],
            'score'    : []
        }

        # get dictionary of dna sequences for all ppms
        dna_seqs = self.generate_list_dna_seq_for_all_ppms()

        # compute score for each PPM and DNA sequence
        for i,ppm in enumerate(self.TfFamily.get_ppms()):
            for j,seqs in enumerate(dna_seqs):
                for k,seq in enumerate(seqs):
                    d['id'].append( self.TfFamily.get_identifiers()[i] )
                    d['prot_seq'].append( self.TfFamily.get_prot_sequences()[i] )
                    d['ppm'].append(ppm)
                    d['dna_seq'].append(seq)
                    if i == j:
                        d['label'].append(1)
                    else:
                        d['label'].append(0)
                    # seq = self._sequence_to_array(seq) Uncomment this line once sklearn encoder has been fixed.
                    one_hot = self._onehote(seq)
                    d['score'].append( self._convolve(ppm, one_hot).max() )
        data = pd.DataFrame.from_dict(d)
        return data


    def generate_list_dna_seq_for_all_ppms(self):
        """ For each PPM, generate a list of random DNA sequences enriched by the motif. 

        Output
        ------
        dna_seqs (list of list of str)   : List of list of DNA sequences 
        """
        dna_seqs = []
        for ppm in self.TfFamily.get_ppms():
            tmp = []
            for _ in range(self.n):
                dna_seq = self._generate_random_dna_seq(self.l)
                dna_seq = self._enrich_dna_seq_with_ppm(dna_seq, ppm)
                tmp.append(dna_seq)
            dna_seqs.append(tmp)
        return dna_seqs


    def get(self):
        return self.data


    def test_dna_seq_generator(self, ppm, l=100):
        random_seq = self._generate_random_dna_seq(l)
        enrich_seq = self._enrich_dna_seq_with_ppm(random_seq, ppm)
        random_one_hot = self._onehote(random_seq)
        enrich_one_hot = self._onehote(enrich_seq)
        random_score = self._convolve(ppm, random_one_hot)
        enrich_score = self._convolve(ppm, enrich_one_hot)
        return random_score, enrich_score


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
    def _onehote(seq):
        seq2=list()
        mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0., 0., 1., 0.], "T":[0., 0., 0., 1.]}
        for i in seq:
            seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.]) 
        return np.stack(seq2).transpose()


    # TODO: Protein encoder that takes as input a sequences array and returns an encoded protein <- actually, should we do it here ? or during model training? 


    @staticmethod
    def _convolve(pwm, one_hot_seq):
        """
        Computes the convolution operation between a one hot encoded sequence and a ppm
        """
        W = pwm.shape[1] # Window size
        L = one_hot_seq.shape[1]-W+1 # Find out about output size
        out = np.zeros(L) # Create output
        for i in range(L): 
            # For each sliding window, compute sum(ppm * seqlet) // convolution
            out[i] = np.multiply(pwm, one_hot_seq[:,i:i+W]).sum()
        return out


""" TODO: rewrite encoder to encode the whole dataset; as it is now this method has some flaws (f.e. ATAA converts to ACAA in one hot matrix)
    @staticmethod
    def _onehote(sequence_array):
        One-hot-encodes a DNA sequence

        sequence (np.array): input DNA sequence in a np array format

        onehot_encoded_seq (matrix): one-hot-encoded sequence

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(sequence_array)
        onehot_encoder = OneHotEncoder(sparse=False, dtype=int, categories=[range(5)])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded = np.delete(onehot_encoded, -1, 1)
        return onehot_encoded.transpose()
"""
