import numpy as np 
import pandas as pd 

class TfFamily:
    """ Class for retrieving the transcription factor information.
    
    """
    
    def __init__(self, ppm_file, prot_file) -> None:
        self.ppm_file = ppm_file
        self.prot_file = prot_file
        self.data = self.parse()
        self.dummy_data = self.generate()

    def parse(self):
        TF_prot_ID, prot = self._parseProt(self.prot_file)
        TF_ppm_ID, ppm = self._parsePPM(self.ppm_file)
        df_1 = pd.DataFrame({"TF_id":[e.split(";")[0] for e in TF_ppm_ID], "TF_ppm_id":TF_ppm_ID, "ppm":ppm})
        df_2 = pd.DataFrame({"TF_id":TF_prot_ID, "prot":prot})
        merged = df_1.merge(df_2, on="TF_id")
        return merged

    def get(self):
        return self.data

    def get_ppms(self):
        return [ppm.transpose() for ppm in self.data["ppm"].values]

    def get_pwms(self):
        return [ self._get_pwm_from_ppm(ppm.transpose()) for ppm in self.data["ppm"].values]

    def get_identifiers(self):
        return self.data["TF_ppm_id"].values

    def get_prot_sequences(self):
        return self.data["prot"].values

    def get_dummy_ppms(self):
        return self.dummy_ppms
    
    def get_dummy_prots(self):
        return self.dummy_prots

    def generate(self):
        prot1 = "RGPRTTIKAKQLETLKXXXXXAAFAATPKPTRHIREQLAQETGLNMRVIQVWFQNRR"
        prot2 = "RGPRTTIKAKQLETLKAAFAATPKPTRHIREQLAQETGLNMRVIYYYYYNRRSKERR"
        ppm1 = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]).T.astype(float)
        ppm2 = np.array([[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]).T.astype(float)

        self.dummy_prots = [prot1, prot2]
        self.dummy_ppms = [ppm1, ppm2]


    @staticmethod
    def _get_pwm_from_ppm(ppm, bg = np.array([0.25,0.25,0.25,0.25]), infval= -5000):
        """
        Converts a ppm into a pwm given a background.

        Params
        ------
        ppm     (numpy float array) : Position probability matrix. rows = ACGT, cols = positions
        bg      (numpy float array) : Vector of background probabilities (alphabetical order)
        infval  (float)             : Value used instead of -infinite 
        
        Output
        ------
        pwm     (numpy float array) : Position wight matrix. rows = ACGT, cols = positions
        """

        # compute pwm
        m = np.log2(ppm.T/bg)
        
        # substitute infinite values with infval 
        m[np.isinf(m)] = infval
        
        return m.T

    @staticmethod
    def _parseProt(prot_file):
        """
        helper for parsing a protein file as defined in README.md 
        """

        prot_array = pd.read_csv(prot_file, sep="\t")
        return prot_array["TF_ID"].values, prot_array["Protein_seq"].values

    @staticmethod
    def _parsePPM(ppm_file):
        """
        Helper function to read the ppm files 
        """
        with open(ppm_file, 'r') as ppm_f:
            # the joint TF id and motif id truly unique
            ppm_id = None
            ppm_array = []
            # the real pbm with values
            ppm_line = None
            find_ppm_lines = None
            ppm_list = []
            tmp_list = []
            for line in ppm_f:
                # Be aware that the following condition on the line is note present before the motif line (it should) the pbm_id will be made of more than 2 fields    
                if len(line.split()) == 2 and line.split()[0] == 'TF':
                    # Attention following ID might not be unique because one sequence can have mutliple motifs
                    tf_id = line.split()[1]
                    ppm_id = tf_id + ';'                
                elif len(line.split()) == 2 and line.split()[0] == 'Motif':
                    motif_id = line.split()[1]
                    ppm_id = ppm_id + motif_id
                    ppm_array.append(ppm_id)
                # Start of PPM value lines, following is the header
                # it should be like this :
                # Pos	A	C	G	T
                elif len(line.split()) == 5 and line.split()[0] == 'Pos':
                    find_ppm_lines = True
                elif line == '\n' and find_ppm_lines:
                    # Tell the script we have passed to new TF momtif so no need to look for ppm lines
                    find_ppm_lines = False
                    ppm_list.append(np.array(tmp_list))
                    tmp_list = [] # resetting temporary list
                elif find_ppm_lines:
                    #pos = line.split()[0]
                    tmp_list.append([ float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4]) ])
                
            # print(ppm_array)
            # print(ppm_list)

            return ppm_array, ppm_list
                
                

        


