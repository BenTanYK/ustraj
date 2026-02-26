"""Functions to analyse US trajectory files"""

import os as _os

import MDAnalysis as _mda
import numpy as _np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import (
    HydrogenBondAnalysis as _HBA,
)
from MDAnalysis.analysis.rdf import InterRDF as _InterRDF
from MDAnalysis.lib.distances import distance_array as _distance_array
from MDAnalysis.transformations.nojump import NoJump as _NoJump
from MDAnalysis.analysis import rms as _rms
from tqdm import tqdm as _tqdm

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
    CV0 : float
        The US bias minimum
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
    CV0 : float
        The US bias minimum
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

def calc_RMSD(
    resid_selection,
    prmtop_filepath,
    free_energy_step,
    restraint_type,
    dof,
    run_number,
    CV0,
    step=1,
    selection_str='backbone',
    u_ref=None
):
    """
    Calculate the RMSD for a region of interest

    Parameters
    ----------
    resid_selection : np.ndarray
        Selection of residue indices to include in the RMSD calculation
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
    CV0 : float
        The US bias minimum
    step : int, optional, default=1
        Step to use when analysing the frames, makes analysis quicker
    selection_str : str, optional, default='backbone'
        Atoms to include during alignment and RMSD calculation
    u_ref : mda.Universe object, optional, default=None
        Reference universe to use for alignment and RMSD calculation.
        If 'None' is specified, the first frame is used

    Returns
    -------
    time : np.ndarray
        Time in ns
    rmsd: np.ndarray
        RMSD as a timeseries
    """
    dcd_filepath = _get_filepath(free_energy_step, restraint_type, dof, run_number, CV0)

    # Check provided topology file exists
    if not _os.path.exists(prmtop_filepath):
        raise FileNotFoundError(f"No topology file found at '{prmtop_filepath}'!")

    u = _mda.Universe(prmtop_filepath, dcd_filepath)

    # Add specific residue indices to selection string
    selection_str += " and resid " + " ".join(str(r) for r in resid_selection)

    selection = u.select_atoms(selection_str)

    if u_ref is not None:
        ref = u_ref.select_atoms(selection_str)
        if ref.n_atoms != selection.n_atoms:
            raise ValueError(
                "Reference and mobile selections have different number of atoms."
            )
    else:
        ref = selection

    R = _rms.RMSD(selection, ref,
    superposition=True)

    R.run(step=step)

    rmsd = R.rmsd.T # transpose

    return rmsd[1]/1000, rmsd[2] # time in ns, rmsd

def calc_closest_distance(
    resids0,
    resids1,
    prmtop_filepath,
    free_energy_step,
    restraint_type,
    dof,
    run_number,
    CV0,
    selection_str='name CA'
):
    """
    Calculate closest distance between two sets of residues

    Parameters
    ----------
    resids0 : np.ndarray
        Residue group 0
    resids1 : np.ndarray
        Residue group 1
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
    CV0 : float
        The US bias minimum
    run_number : int
        The desired replicate
    selection_str : str, optional, default='name CA'
        Atoms to include in the distance calculations

    Returns
    -------
    time : np.ndarray
        Time in ns
    distance: np.ndarray
        Closest distance between the two selections over time
    """

    dcd_filepath = _get_filepath(free_energy_step, restraint_type, dof, run_number, CV0)

    # Check provided topology file exists
    if not _os.path.exists(prmtop_filepath):
        raise FileNotFoundError(f"No topology file found at '{prmtop_filepath}'!")

    u = _mda.Universe(prmtop_filepath, dcd_filepath)
    transformation = _NoJump() # Account for PBC jumps
    u.trajectory.add_transformations(transformation)

    sel_str0 = f"({selection_str}) and resid " + " ".join(str(r) for r in resids0)
    sel_str1 = f"({selection_str}) and resid " + " ".join(str(r) for r in resids1)

    sel0 = u.select_atoms(sel_str0)
    sel1 = u.select_atoms(sel_str1)

    if sel0.n_atoms == 0:
        raise ValueError('Selection 0 is empty!')
    if sel1.n_atoms == 0:
        raise ValueError('Selection 1 is empty!')

    distances = []
    time = []

    for ts in _tqdm(u.trajectory, desc='Frames analysed'):
        # Compute all pairwise distances (residue atoms vs ligand atoms)
        d = _distance_array(sel0.positions, sel1.positions)

        # Store the minimum distance for this frame
        distances.append(d.min())
        time.append(ts.time)

    return 1e-3*_np.array(time), _np.array(distances)

def calc_hbond_distance(
    donor_id,
    acceptor_id,
    prmtop_filepath,
    free_energy_step,
    restraint_type,
    dof,
    run_number,
    CV0,
    donor_selection=None,
    acceptor_selection=None
):
    """
    Calculate closest distance between two sets of residues

    Parameters
    ----------
    donor_id : int
        Index of the donor residue
    acceptor_id : int
        Index of the acceptor residue
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
    CV0 : float
        The US bias minimum
    donor_selection : str, optional, default=None
        Additional selection string, e.g. 'not name H'
    acceptor_selection : str, optional, default=None
        Additional selection string, e.g. 'not name O'

    Returns
    -------
    time : np.ndarray
        Time in ns
    distance: np.ndarray
        Closest distance between the donor and acceptor atoms over time
    """
    dcd_filepath = _get_filepath(free_energy_step, restraint_type, dof, run_number, CV0)

    # Check provided topology file exists
    if not _os.path.exists(prmtop_filepath):
        raise FileNotFoundError(f"No topology file found at '{prmtop_filepath}'!")

    u = _mda.Universe(prmtop_filepath, dcd_filepath)
    transformation = _NoJump() # Account for PBC jumps
    u.trajectory.add_transformations(transformation)

    donor_sel = f"resid {donor_id} and (name H*)"
    acceptor_sel = f"resid {acceptor_id} and (name O* or name N* or name S*)"

    if donor_selection is not None:
        donor_sel += f" and ({donor_selection})"
    if acceptor_selection is not None:
        acceptor_sel += f" and ({acceptor_selection})"

    donors = u.select_atoms(donor_sel)
    acceptors = u.select_atoms(acceptor_sel)

    if donors.n_atoms == 0:
        raise ValueError('Donor selection is empty!')
    if acceptors.n_atoms == 0:
        raise ValueError('Acceptor selection is empty!')

    distances = []
    time = []

    for ts in _tqdm(u.trajectory, desc='Frames analysed'):
        # Compute all pairwise distances (residue atoms vs ligand atoms)
        d = _distance_array(donors.positions, acceptors.positions)

        # Store the minimum distance for this frame
        distances.append(d.min())
        time.append(ts.time)

    return 1e-3*_np.array(time), _np.array(distances)
