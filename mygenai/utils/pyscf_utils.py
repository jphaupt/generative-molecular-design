from pyscf import gto, dft
from pyscf import grad
import numpy as np
HARTREE2EV = 27.211386246

def get_homo_lumo_gap(atomic_numbers, positions, dft_xc="B3LYP", basis="6-31G(2df,p)"):
    """
    Calculate the HOMO-LUMO gap for a molecular system using DFT.
    Parameters:
        atomic_numbers (numpy list of int): A list of atomic numbers representing the elements in the molecule.
        positions (numpy list of list of float): A list of 3D coordinates (x, y, z) for each atom in the molecule.
        dft_xc (str, optional): The exchange-correlation functional to use for DFT calculations.
            Defaults to "B3LYP".
        basis (str, optional): The basis set to use for the calculations.
            Defaults to "6-31G(2df,p)".
    Returns:
        SCF: PySCF DFT object for the calculation.
        float: The HOMO-LUMO gap in eV.
    Raises:
        ValueError: If the input data is invalid or the calculation fails.
    """
    mol = gto.Mole()
    mol.atom = [(int(atomic_numbers[i]), positions[i]) for i in range(len(atomic_numbers))]
    mol.basis = basis
    mol.build()

    # Run DFT
    mf = dft.RKS(mol)
    mf.xc = dft_xc
    mf.kernel()

    # get HOMO-LUMO
    mo_energies = mf.mo_energy  # in Hartree
    homo_idx = mol.nelectron // 2 - 1
    lumo_idx = homo_idx + 1

    homo_lumo_gap = (mo_energies[lumo_idx] - mo_energies[homo_idx]) * HARTREE2EV

    return mf, homo_lumo_gap
