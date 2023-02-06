import numpy as np 
import pandas as pd 

class TfFamily:
    """ Class for retrieving the transcription factor information.
    
    """
    
    def __init__(self, pwm_file, prot_file) -> None:
        self.pwm_file = pwm_file
        self.prot_file = prot_file
        self.data = self.parse()

    def parse(self):
        TF_prot_ID, prot = self._parseProt(self.prot_file)
        TF_pwm_ID, pwm = self._parsePWM(self.pwm_file)
        df_1 = pd.DataFrame({"TF_id":[e.split(";")[0] for e in TF_pwm_ID], "TF_pwm_id":TF_pwm_ID, "pwm":pwm})
        df_2 = pd.DataFrame({"TF_id":TF_prot_ID, "prot":prot})
        merged = df_1.merge(df_2, on="TF_id")
        return merged

    def get(self):
        return self.data

    @staticmethod
    def _parseProt(prot_file):
        """
        helper for parsing a protein file as defined in README.md 
        """

        prot_array = pd.read_csv(prot_file, sep="\t")
        return prot_array["TF_ID"].values, prot_array["Protein_seq"].values

    @staticmethod
    def _parsePWM(pwm_file):
        with open(pwm_file, 'r') as pwm_f:
            # the joint TF id and motif id truly unique
            pbm_id = None
            pbm_array = []
            # the real pbm with values
            pbm_line = None
            find_pbm_lines = None
            pbm_list = []
            tmp_list = []
            for line in pwm_f:
                # Be aware that the following condition on the line is note present before the motif line (it should) the pbm_id will be made of more than 2 fields    
                if len(line.split()) == 2 and line.split()[0] == 'TF':
                    # Attention following ID might not be unique because one sequence can have mutliple motifs
                    tf_id = line.split()[1]
                    pbm_id = tf_id + ';'                
                elif len(line.split()) == 2 and line.split()[0] == 'Motif':
                    motif_id = line.split()[1]
                    pbm_id = pbm_id + motif_id
                    pbm_array.append(pbm_id)
                # Start of PBM value lines, following is the header
                # it should be like this :
                # Pos	A	C	G	T
                elif len(line.split()) == 5 and line.split()[0] == 'Pos':
                    find_pbm_lines = True
                elif line == '\n' and find_pbm_lines:
                    # Tell the script we have passed to new TF momtif so no need to look for pbm lines
                    find_pbm_lines = False
                    pbm_list.append(np.array(tmp_list))
                    tmp_list = [] # resetting temporary list
                elif find_pbm_lines:
                    #pos = line.split()[0]
                    tmp_list.append([ float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4]) ])
                
            #print(pbm_array)
            #print(pbm_list)

            return pbm_array, pbm_list
                
                

        


