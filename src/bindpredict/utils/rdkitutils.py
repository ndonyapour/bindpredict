import numpy as np
import warnings
import collections.abc
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs


class PhysiochemicalProp(collections.abc.Mapping):
    def __init__(self, molecule):
        self.molecule = molecule

        setattr(self,
                'LogP',
                Descriptors.MolLogP(self.molecule))
        setattr(self, 'MolWt',
                Descriptors.MolWt(self.molecule))

        setattr(self, 'PSA',
                Descriptors.TPSA(self.molecule))

        setattr(self, 'NumHAcceptors',
                Descriptors.NumHAcceptors(self.molecule))

        setattr(self, 'NumHDonors',
                Descriptors.NumHDonors(self.molecule))

        setattr(self, 'NumRotatableBonds',
                Descriptors.NumRotatableBonds(self.molecule))

        setattr(self, 'RingCount',
                Descriptors.RingCount(self.molecule))

    def __iter__(self):
        yield 'LogP'
        yield 'MolWt'
        yield 'PSA'
        yield 'NumHAcceptors'
        yield 'NumHDonors'
        yield 'NumRotatableBonds'
        yield 'RingCount'

    def __len__(self):
        return 7

    def __getitem__(self, item):
        return getattr(self, item)


class MolFingerprints(object):
    def __init__(self, molecule):
        self.molecule = molecule

    @property
    def ECPF4(self):
        fp = AllChem.GetMorganFingerprintAsBitVect(self.molecule,
                                                   2,
                                                   nBits=1024)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr

    @property
    def MACCS(self):
        return np.array(MACCSkeys.GenMACCSKeys(self.molecule))


def canonize_smile(smile, canonize=True):
    new_smile = None

    try:
        new_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile,
                                                        sanitize=canonize))

    except:

        warnings.warn(smile + ' can not be canonized:'
                      'nvalid SMILES string!')


    return new_smile


# standardize
def standardize(mol_smiles):
    flags = []
    for smiles in mol_smiles:
        canonized_smile = canonize_smile(smiles)

        if canonized_smile:
            flags.append('Yes')
        else:
            flags.append('No')

    return flags


def smiles_to_mol(mol_smiles):
    mols = []
    for smiles in mol_smiles:
        mols.append(Chem.MolFromSmiles(smiles))
    return mols
