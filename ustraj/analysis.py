"""Functions to analyse US trajectory files"""

import os as _os

import MDAnalysis as _mda
import numpy as _np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
    HydrogenBondAnalysis as _HBA,
)
from MDAnalysis.analysis.rdf import InterRDF as _InterRDF


def _get_filepath(free_energy_step, restraint_type, dof, run_number, CV0):
    """Obtain the relative dcd filepath for a given file"""

    CV0 = _np.round(CV0, 3)

    if free_energy_step == "separation":
        filepath = (
            f"{free_energy_step}/results/{restraint_type}_RMSD/run{run_number}/{CV0}.dcd"
        )
    elif free_energy_step == "Boresch" or free_energy_step == "RMSD":
        filepath = (
            f"{free_energy_step}/results/{restraint_type}_RMSD/{dof}/run{run_number}/{CV0}.dcd"
        )
    else:
        raise ValueError("Select 'separation', 'Boresch' or 'RMSD' as free_energy_step")

    # Check that the trajectory file exists
    if not _os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory file '{filepath}' does not exist!")

    return filepath


def count_hbonds(
    residue_index,
    prmtop_filepath,
    free_energy_step,
    restraint_type,
    dof,
    run_number,
    CV0,
    sidechain_only=True,
    step=10,
    solvent_only=False,
):
    """
    Calculate the number of hydrogen bonds formed to a specific residue during
    the course of a certain trajectory file.

    Parameters
    ----------
    residue_index : int
        Residue of interest for hydrogen bonding analysis

    prmtop_filepath : str
        The filepath for the topology file corresponding to the desired trajectory

    free_energy_step : str
        The desired type of US simulation. The options are 'separation',
        'RMSD' or 'Boresch'.

    restraint_type : str
        The type of RMSD restraints used. The options are 'CA' or 'backbone'.

    dof : str
        The degree of freedom. For free_energy_step='Boresch', the options are
        'thetaA', 'thetaB', 'phiA', 'phiB', 'phiC'. For free_energy_step='RMSD',
        the options specify bound or bulk state, as well as the restrain order.

    run_number : int
        The desired replicate

    sidechain_only : bool, optional, default=True
        Only use the sidechain of the residue

    step : int, optional, default=10
        Step to use when analysing the frames, makes analysis quicker

    solvent_only: bool, optional, default=False
        Only consider H-bonding to solvent

    Returns
    -------
    donor_data: dict
        Dictionary containing information about the residue of interest being
        involved as a hydrogen bond donor.

    acceptor_data: dict
        Dictionary containing information about the residue of interest being
        involved as a hydrogen bond acceptor.

    """
    dcd_filepath = _get_filepath(free_energy_step, restraint_type, dof, run_number, CV0)

    # Check provided topology file exists
    if not _os.path.exists(prmtop_filepath):
        raise FileNotFoundError(f"No topology file found at '{prmtop_filepath}'!")

    u = _mda.Universe(prmtop_filepath, dcd_filepath)

    water_sel = "resname WAT"
    res_id = f"resid {residue_index}"

    # residue atoms to consider (sidechain vs whole residue)
    res_sel = f"({res_id}) and not backbone" if sidechain_only else f"({res_id})"

    partner_sel = (
        f"not ({res_id})"  # exclude the whole residue regardless of sidechain_only
    )

    # Residue acting as donor
    hydrogens0 = f"({res_sel}) and name H*"
    acceptors0 = f"({partner_sel}) and (name O* or name N* or name S*)"

    # Residue acting as acceptor
    hydrogens1 = f"({partner_sel}) and name H*"
    acceptors1 = f"({res_sel}) and (name O* or name N* or name S*)"

    if solvent_only:
        acceptors0 += f" and ({water_sel})"
        hydrogens1 += f" and ({water_sel})"

    # Identify possible donor interactions
    h0 = _HBA(
        u,
        hydrogens_sel=hydrogens0,
        acceptors_sel=acceptors0,
    )
    h0.run(start=0, stop=None, step=step, verbose=True)

    # Identify possible acceptor interactions
    h1 = _HBA(
        u,
        hydrogens_sel=hydrogens1,
        acceptors_sel=acceptors1,
    )
    h1.run(start=0, stop=None, step=step, verbose=True)

    # Find simulation time
    t_final = _np.round(u.trajectory[-1].time / 1000, 4)
    time = _np.linspace(0, t_final, len(h0.count_by_time()))

    donor_data = {
        "Time (ns)": time,
        "Count by time": h0.count_by_time(),
        "Count by ID": h0.count_by_ids(),
        "Count by type": h0.count_by_type(),
    }

    acceptor_data = {
        "Time (ns)": time,
        "Count by time": h1.count_by_time(),
        "Count by ID": h1.count_by_ids(),
        "Count by type": h1.count_by_type(),
    }

    return donor_data, acceptor_data


def calc_solvent_rdf(
    atom_index,
    prmtop_filepath,
    free_energy_step,
    restraint_type,
    dof,
    run_number,
    CV0,
    step=10,
):
    """
    Calculate the solvent RDF around an atom of interest

    Parameters
    ----------
    atom_index : int
        Atom used to compute the RDF

    prmtop_filepath : str
        The filepath for the topology file corresponding to the desired trajectory

    free_energy_step : str
        The desired type of US simulation. The options are 'separation',
        'RMSD' or 'Boresch'.

    restraint_type : str
        The type of RMSD restraints used. The options are 'CA' or 'backbone'.

    dof : str
        The degree of freedom. For free_energy_step='Boresch', the options are
        'thetaA', 'thetaB', 'phiA', 'phiB', 'phiC'. For free_energy_step='RMSD',
        the options specify bound or bulk state, as well as the restrain order.

    run_number : int
        The desired replicate

    step : int, optional, default=10
        Step to use when analysing the frames, makes analysis quicker

    Returns
    -------
    distance : np.ndarray
        Radial separation distance in Angstroms

    rdf: np.ndarray
        Normalised RDF around the specified atom

    """
    dcd_filepath = _get_filepath(free_energy_step, restraint_type, dof, run_number, CV0)

    # Check provided topology file exists
    if not _os.path.exists(prmtop_filepath):
        raise FileNotFoundError(f"No topology file found at '{prmtop_filepath}'!")

    u = _mda.Universe(prmtop_filepath, dcd_filepath)

    selected_atom = u.select_atoms(f"index {atom_index}")
    solvent = u.select_atoms("resname WAT")

    rdf = _InterRDF(selected_atom, solvent)
    rdf.run(step=step, verbose=True)

    return rdf.bins, rdf.rdf
