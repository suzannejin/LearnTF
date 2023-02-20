import numpy as np
import pandas as pd
import random
import src.utils.preProcessing as pre


class SimulatedData:
    """

    This class constructs a simulation dataset from raw data.


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
        self.dummy_data = self.simulate_dummy_data()


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

        # get list of dna sequences
        dna_seqs = self.generate_list_dna_seq_for_all_ppms()
        random_seqs = []
        for _ in range(self.n):
            random_seqs.append(self._generate_random_dna_seq(self.l))

        # compute score for each PPM and DNA sequence
        pwms = self.TfFamily.get_pwms()
        for i,ppm in enumerate(self.TfFamily.get_ppms()):
            # compute scores for enriched sequences
            for j,seqs in enumerate(dna_seqs):
                for seq in seqs:
                    d['id'].append( self.TfFamily.get_identifiers()[i] )
                    d['prot_seq'].append( self.TfFamily.get_prot_sequences()[i] )
                    d['ppm'].append(ppm)
                    d['dna_seq'].append(seq)
                    if i == j:
                        d['label'].append(1)
                    else:
                        d['label'].append(0)
                    one_hot = pre.encode_dna(seq)
                    d['score'].append( self._convolve(pwms[i], one_hot).max() )
            # compute scores for additional random sequences
            for seq in random_seqs:
                d['id'].append( self.TfFamily.get_identifiers()[i] )
                d['prot_seq'].append( self.TfFamily.get_prot_sequences()[i] )
                d['ppm'].append(ppm)
                d['dna_seq'].append(seq)
                d['label'].append(-1)
                one_hot = pre.encode_dna(seq)
                d['score'].append( self._convolve(pwms[i], one_hot).max() )
        data = pd.DataFrame.from_dict(d)
        return data
        

    def simulate_dummy_data(self, n=100, l=100):
        """ Simulate DNA sequences and score them to each PPM for the entire TF family.

        Output
        ------
        scores  (numpy array of float) : List scores matching a DNA sequence with a PPM

        """
        
        base_sequence = "T"*l

        # initialize data dictionary
        d = {
            'id'       : [],
            'prot_seq' : [],
            'ppm'      : [],
            'dna_seq'  : [],
            'label'    : []
        }


        # compute score for each PPM and DNA sequence
        for i,ppm_1 in enumerate(self.TfFamily.get_dummy_ppms()):
            for j,ppm_2 in enumerate(self.TfFamily.get_dummy_ppms()):
                for k in range(n):

                    d['id'].append(str("prot_") + str(i) + str(j) + str("seq_") + str(k) )
                    d['prot_seq'].append( self.TfFamily.get_dummy_prots()[i] )
                    d['ppm'].append(ppm_1)

                    seq = self._enrich_dna_seq_with_ppm(base_sequence, ppm_2)
                    d['dna_seq'].append(seq)
                    if i == j:
                        d['label'].append(1)
                    else:
                        d['label'].append(0)

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
        dna_seq = ''.join(random.choice('ACGT') for _ in range(l))
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

        # Check that the requested dna length does not crach the random choice
        assert (len(dna_seq) > npos), "The sequence length requested is too short with respect to ppm length "

        # modify DNA sequence    
        start = random.choice(range(len(dna_seq)-npos))
        end   = start + npos
        enriched_seq = list(dna_seq)
        enriched_seq[start:end] = chunk
        enriched_seq = ''.join(enriched_seq)

        return enriched_seq


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
