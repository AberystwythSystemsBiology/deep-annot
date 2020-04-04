from keras.utils import Sequence
import pyisopach
import numpy as np
import random
from pyisopach.periodic_table import get_periodic_table

rules = {
    "negative" : {
        "[M1-.]1-": [{"add": {}}, -1, -0],
        "[3M-H]1-": [{"remove": {"H": 1}, "multiply": 3}, -1, 1],
        "[2M+Hac-H]1-": [{"add": {"C": 2, "H": 3, "O": 2}, "multiply": 2}, -1, 1],
        "[2M+FA-H]1-": [{"add": {"C": 1, "H": 1, "O": 2}, "multiply": 2}, -1, 1],
        "[2M-H]1-": [{"remove": {"H": 1}, "multiply": 2}, -1, 1],
        "[M+TFA-H]1-": [{"add": {"C": 2, "O": 2, "F": 3}}, -1, 1],
        "[M+Hac-H]1-": [{"add": {"C": 2, "H": 3, "O": 2}}, -1, 1],
        "[M+FA-H]1-": [{"add": {"C": 1, "H": 1, "O": 2}}, -1, 1],
        "[M-H]1-": [{"remove": {"H": 1}}, -1, 1],
        "[M-2H]2-": [{"remove": {"H": 2}}, -2, 2],
        "[M-3H]3-": [{"remove": {"H": 3}}, -3, 3],
        "[2M+Na-2H]1-": [{"add": {"Na": 1}, "remove": {"H": 2}, "multiply": 2}, -1, 1],
        "[M+K-2H]1-": [{"add": {"K": 1}, "remove": {"H": 2}}, -1, 1],
        "[M+Na-2H]1-": [{"add": {"Na": 1}, "remove": {"H": 2}}, -1, 1],
        "[M+Br]1-": [{"add": {"Br": 1}}, -1, 1],
        "[M+Cl]1-": [{"add": {"Cl": 1}}, -1, 1]
    },
    "positive": []
}

class DataGenerator(Sequence):

    def __init__(self, num_molecules:int = 100, polarity: str = "negative", max_length: int=10, batch_size: int=32, shuffle: bool=True):
        

        self.num_molecules = num_molecules
        self.rules = rules[polarity]
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.generate_molecules()
        self.on_epoch_end()


    def generate_molecules(self) -> None:
        elements = list(get_periodic_table().keys())

        molecules = []

        for i in range(self.num_molecules):
            molecule_dict = {}

            for j in range(random.randint(5, 19)):

                e = random.choice(elements)
                n_element = random.randint(1, 42)

                if e not in molecule_dict:
                    molecule_dict[e] = n_element
                else:
                    molecule_dict[e] += n_element

            m = "".join(["".join([e, str(c)]) for e, c in molecule_dict.items()])

            molecules.append(m)
        

        self.molecules = np.array(molecules)

        

    def __len__(self) -> int:
        return int(np.floor(len(self.molecules) / self.batch_size))

    
    def on_epoch_end(self):
        self.indexes = np.arange(self.num_molecules)
        if self.shuffle:
            np.random.shuffle(self.indexes)



    def __getitem__(self, index: int):
        indexes = self.indexes[index * self.batch_size : self.batch_size * (index + 1)]

        molecules_tmp = self.molecules[indexes]

        X, y = self.generate(molecules_tmp)

    def generate(self, molecules_tmp: list):
        d = [self.get_distributions(x) for x in molecules_tmp]
        return X, y


    

    def get_distributions(self, molecule):


        def _generate_noise(masses, intensities):
            noise = np.random.normal(0, 100, int(intensities.shape[0] * random.random()))

            print(noise)

        isopach_mol = pyisopach.Molecule(molecule)

        adduct = random.choice(list(self.rules.keys()))
        rule_dict, charge, electrons  = self.rules[adduct]

        structure_dict = isopach_mol._structure_dict

        if "multiply" in rule_dict:
            for element, amount in structure_dict.items():
                structure_dict[element] = amount * rule_dict["multiply"]
        if "add" in rule_dict:
            for element, amount in rule_dict["add"].items():
                if element not in structure_dict:
                    structure_dict[element] = amount
                else:
                    structure_dict[element] += amount
        if "remove" in rule_dict:
            for element, amount in rule_dict["remove"].items():
                # NOTE: Need to clarify this, maybe.
                if element in structure_dict:
                    structure_dict[element] -= amount           
            

        isopach_mol._structure_dict = structure_dict

        masses, intensities = isopach_mol.isotopic_distribution(electrons = electrons, charge = charge)

        _generate_noise(masses, intensities)

        return masses, intensities, adduct