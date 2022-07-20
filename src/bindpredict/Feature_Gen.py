import numpy as np
from collections import defaultdict

from bindpredict.utils.rdkitutils import PhysiochemicalProp, MolFingerprints


class Featurizer(object):
    def __init__(self, prop_names=['Physicochemical', 'ECPF4', 'MACCS']):
        self.prop_names = prop_names

    def generate(self, molecules):
        features = defaultdict(list)

        if 'Physicochemical' in self.prop_names:
            for mol in molecules:
                physchem = PhysiochemicalProp(mol)
                features['LogP'].append(physchem.LogP)
                features['MolWt'].append(physchem.MolWt)
                features['PSA'].append(physchem.PSA)
                features['NumHAcceptors'].append(physchem.NumHAcceptors)
                features['NumHDonors'].append(physchem.NumHDonors)
                features['NumRotatableBonds'].append(physchem.NumRotatableBonds)
                features['RingCount'].append(physchem.RingCount)

        if 'ECPF4' in self.prop_names:
            for mol in molecules:
                features['ECPF4'].append(MolFingerprints(mol).ECPF4)

        if 'MACCS' in self.prop_names:
            for mol in molecules:
                features['MACCS'].append(MolFingerprints(mol).MACCS)

        return features


def create_X_Y(df, selected_features, y_key):

    data = []
    num_rows = df.shape[0]
    for key in selected_features:
        x_key = []
        for idx in range(num_rows):
            x_key.append(np.array(df[key].values[idx]))

        x_key  = np.array(x_key)
        if len(np.array(x_key).shape) == 1:
            x_key = x_key.reshape(-1, 1)

        data.append(x_key)
    X = np.concatenate(data, axis=1)

    Y = df[y_key].values

    return X, Y
