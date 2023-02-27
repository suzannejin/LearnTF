import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Transform sequence into array
def sequence_to_array(sequence):
    sequence = sequence.lower()
    sequence_array = np.array(list(sequence))
    return sequence_array.reshape(-1, 1)

# OneHot encoder for dna sequences
def encode_dna(dna, alphabet="acgt"):
    dna = sequence_to_array(dna)
    encoder = OneHotEncoder(categories=[list(alphabet)])
    return encoder.fit_transform(dna).toarray().T

# OneHot encoder for protein sequences
def encode_protein(protein, alphabet="acdefghiklmnpqrstvwxy"):
    protein = sequence_to_array(protein)
    encoder = OneHotEncoder(categories=[list(alphabet)])
    return encoder.fit_transform(protein).toarray().T



