import os
import os.path as osp
import numpy as np

from openbabel import pybel

from geomm.superimpose import superimpose
from geomm.rmsd import calc_rmsd


def convert_sdf_pdb(sdf_files_path, output_path):
    """ Converts sdf files in the input folder to pdb
    """
    sdf_files = [f for f in os.listdir(sdf_files_path)
                 if f.endswith(".sdf")]

    for sdf_file in sdf_files:
        sdf_file_path = osp.join(sdf_files_path, sdf_file)
        if osp.isfile(sdf_file_path):

            mol =  list(pybel.readfile("sdf", sdf_file_path))[0]
            pdb_file_path = osp.join(output_path,
                                     osp.splitext(sdf_file)[0]+'.pdb')
            mol.write('pdb', pdb_file_path)
        else:
            print(f"Can not open the file {sdf_file}")


def calc_rmsds(ref_traj, pose_traj, align_idxs, rmsd_idxs):
    """Superimpose a trajectory uisng align_idxs and then calculates
    RMSEs of aligned structres based on rmsd_idxs
    """

    ref_coords = ref_traj.xyz[0]
    sup_coords = []
    rmsds = []
    for pose_coords in pose_traj.xyz:
        pose_sup_coords, _, _ = superimpose(ref_coords,
                                            pose_coords,
                                            align_idxs)
        rmsds.append(calc_rmsd(ref_coords[rmsd_idxs],
                               pose_sup_coords[rmsd_idxs]))

    #conver to Angstrom
    return np.array(rmsds)*10


def calc_traj_to_traj_rmsds(first_traj, second_traj, align_idxs, rmsd_idxs):
    """Superimpose a trajectory uisng align_idxs and then calculates
    RMSEs of aligned structres based on rmsd_idxs
    """

    rmsds = []
    for idx in range(first_traj.n_frames):
        pose_sup_coords, _, _ = superimpose(first_traj.xyz[idx],
                                            second_traj.xyz[idx],
                                            align_idxs)
        rmsds.append(calc_rmsd(first_traj.xyz[idx][rmsd_idxs],
                     pose_sup_coords[rmsd_idxs]))

    #conver to Angstrom
    return np.array(rmsds)*10
