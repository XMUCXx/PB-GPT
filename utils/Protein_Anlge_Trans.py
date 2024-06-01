"""
Code to convert from angles between residues to XYZ coordinates. 
"""
import functools
import gzip
import os
import logging
import glob
from collections import namedtuple, defaultdict
from itertools import groupby
from typing import *
import warnings

from biopandas.pdb import PandasPdb

import numpy as np
import pandas as pd

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.sequence import ProteinSequence

from . import nerf

EXHAUSTIVE_ANGLES = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
EXHAUSTIVE_DISTS = ["0C:1N", "N:CA", "CA:C"]

MINIMAL_ANGLES = ["phi", "psi", "omega"]
MINIMAL_DISTS = []
def extract_backbone(fname:str):
        def calc_dihedral(p): #[p1,p2,p3,p4]
            """Praxeolitic formula
            1 sqrt, 1 cross product"""
            p0 = p[0]
            p1 = p[1]
            p2 = p[2]
            p3 = p[3]

            b0 = -1.0*(p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2

            # normalize b1 so that it does not influence magnitude of vector
            # rejections that come next
            b1 /= np.linalg.norm(b1)

            # vector rejections
            # v = projection of b0 onto plane perpendicular to b1
            #   = b0 minus component that aligns with b1
            # w = projection of b2 onto plane perpendicular to b1
            #   = b2 minus component that aligns with b1
            v = b0 - np.dot(b0, b1)*b1
            w = b2 - np.dot(b2, b1)*b1

            # angle between v and w in a plane is the torsion angle
            # v and w may not be normalized but that's fine since tan is y/x
            x = np.dot(v, w)
            y = np.dot(np.cross(b1, v), w)
            return np.degrees(np.arctan2(y, x))/180*np.pi
        def calc_bond_angle(a,b,c):
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            return np.degrees(angle)/180*np.pi
        
        def calc_distance(a,b):
            return np.linalg.norm(a-b)

        data = PandasPdb().read_pdb(fname)
        df = data.df['ATOM']
        df=df[((df.atom_name == 'N') | (df.atom_name == 'CA') | (df.atom_name == 'C')) & (df.chain_id==df['chain_id'][0]) & (df.alt_loc==df['alt_loc'][0])]
        
        df.index=pd.RangeIndex(0,len(df.index),1,dtype='int64')      
        
        while (len(df.index)-1>=0) and df['atom_name'][len(df.index)-1]!='C':
            df=df.drop(index = [len(df.index)-1],inplace=False)
            df.index=pd.RangeIndex(0,len(df.index),1,dtype='int64')
        
        coords=[] #[{N,Ca,C}]
        #distance=[] #[{N:CA,CA:C,0C:1N}]
        #angles=[] #[{phi,psi,omega,tau,"CA:C:1N","C:1N:1CA"}]
        dist_0C_1N=[]
        dist_N_CA=[]
        dist_CA_C=[]

        angle_psi=[]
        angle_omega=[]
        angle_phi=[{"phi":np.float32(np.nan)}]
        angle_tau=[]
        angle_CA_C_1N=[]
        angle_C_1N_1CA=[]
         
        for i in df.index:
            if i%3==0:
                if df['atom_name'][i]!='N' or df['atom_name'][i+1]!='CA' or df['atom_name'][i+2]!='C':
                    coords=[]
                    break
                #coordinates
                N_coords=[df['x_coord'][i],df['y_coord'][i],df['z_coord'][i]]
                N_coords=np.array(N_coords).astype(np.float32).tolist()

                CA_coords=[df['x_coord'][i+1],df['y_coord'][i+1],df['z_coord'][i+1]]
                CA_coords=np.array(CA_coords).astype(np.float32).tolist()

                C_coords=[df['x_coord'][i+2],df['y_coord'][i+2],df['z_coord'][i+2]]
                C_coords=np.array(C_coords).astype(np.float32).tolist()

                if i+3<len(df.index):
                    N1_coords=[df['x_coord'][i+3],df['y_coord'][i+3],df['z_coord'][i+3]]
                    N1_coords=np.array(N1_coords).astype(np.float32).tolist()
                else:
                    N1_coords=C_coords

                if i+4<len(df.index):
                    CA1_coords=[df['x_coord'][i+4],df['y_coord'][i+4],df['z_coord'][i+4]]
                    CA1_coords=np.array(CA1_coords).astype(np.float32).tolist()
                else:
                    CA1_coords=C_coords

                if i+5<len(df.index):
                    C1_coords=[df['x_coord'][i+5],df['y_coord'][i+5],df['z_coord'][i+5]]
                    C1_coords=np.array(C1_coords).astype(np.float32).tolist()
                else:
                    C1_coords=C_coords

                N=np.array(N_coords)
                CA=np.array(CA_coords)
                C=np.array(C_coords)

                N1=np.array(N1_coords)
                CA1=np.array(CA1_coords)
                C1=np.array(C1_coords)

                #coords.append({"N":N_coords,"CA":CA_coords,"C":C_coords})
                coords.append(N_coords)
                coords.append(CA_coords)
                coords.append(C_coords)

                #edge_len 
                N_CA=calc_distance(N,CA)
                CA_C=calc_distance(CA,C)
                C_N1=calc_distance(C,N1)
                if i+3>len(df.index):
                    C_N1=np.nan
                #distance.append({"N:CA":N_CA,"CA:C":CA_C,"0C:1N":C_N1})
                dist_0C_1N.append({"0C:1N":np.float32(C_N1)})
                dist_N_CA.append({"N:CA":np.float32(N_CA)})
                dist_CA_C.append({"CA:C":np.float32(CA_C)})
                #angles
                if i+3<len(df.index):
                    psi=calc_dihedral([N,CA,C,N1])
                    omega=calc_dihedral([CA,C,N1,CA1])
                    phi=calc_dihedral([C,N1,CA1,C1])
                    tau=calc_bond_angle(N,CA,C)
                    CA_C_1N=calc_bond_angle(CA,C,N1)
                    C_1N_1CA=calc_bond_angle(C,N1,CA1)
                else:
                    psi=np.nan
                    omega=np.nan
                    phi=np.nan
                    tau=calc_bond_angle(N,CA,C)
                    CA_C_1N=np.nan
                    C_1N_1CA=np.nan

                #angles.append({"psi":psi,"omega":omega,"phi":phi,"tau":tau,"CA:C:1N":CA_C_1N,"C:1N:CA":C_1N_1CA})
                angle_psi.append({"psi":np.float32(psi)})
                angle_omega.append({"omega":np.float32(omega)})
                angle_phi.append({"phi":np.float32(phi)})
                angle_tau.append({"tau":np.float32(tau)})
                angle_CA_C_1N.append({"CA:C:1N":np.float32(CA_C_1N)})
                angle_C_1N_1CA.append({"C:1N:1CA":np.float32(C_1N_1CA)})

        #angles
        #distance
        #coords=pd.DataFrame(coords)
        coords=np.array(coords)

        dist_N_CA.append({"N:CA":np.float32(0)})
        dist_CA_C.append({"CA:C":np.float32(0)})

        
        dist_0C_1N=pd.DataFrame(dist_0C_1N)
        dist_N_CA=pd.DataFrame(dist_N_CA[1:])
        dist_CA_C=pd.DataFrame(dist_CA_C[1:])

        angle_psi=pd.DataFrame(angle_psi)
        angle_omega=pd.DataFrame(angle_omega)
        angle_phi=pd.DataFrame(angle_phi[:-1])
        angle_tau=pd.DataFrame(angle_tau[1:])
        angle_CA_C_1N=pd.DataFrame(angle_CA_C_1N)
        angle_C_1N_1CA=pd.DataFrame(angle_C_1N_1CA)
        
        
        my_data=pd.concat([angle_phi,angle_psi,angle_omega,angle_tau,angle_CA_C_1N,angle_C_1N_1CA,dist_0C_1N,dist_N_CA,dist_CA_C],axis=1)
        return coords,my_data
def create_new_chain_nerf(
    out_fname: str,
    dists_and_angles: pd.DataFrame,
    angles_to_set: Optional[List[str]] = None,
    dists_to_set: Optional[List[str]] = None,
    center_coords: bool = True,
) -> str:
    """
    Create a new chain using NERF to convert to cartesian coordinates. Returns
    the path to the newly create file if successful, empty string if fails.
    """
    if angles_to_set is None and dists_to_set is None:
        angles_to_set, dists_to_set = [], []
        for c in dists_and_angles.columns:
            # Distances are always specified using one : separating two atoms
            # Angles are defined either as : separating 3+ atoms, or as names
            if c.count(":") == 1:
                dists_to_set.append(c)
            else:
                angles_to_set.append(c)
        logging.debug(f"Auto-determined setting {dists_to_set, angles_to_set}")
    else:
        assert angles_to_set is not None
        assert dists_to_set is not None

    # Check that we are at least setting the dihedrals
    required_dihedrals = ["phi", "psi", "omega"]
    assert all([a in angles_to_set for a in required_dihedrals])
    nerf_build_kwargs = dict(
        phi_dihedrals=dists_and_angles["phi"],
        psi_dihedrals=dists_and_angles["psi"],
        omega_dihedrals=dists_and_angles["omega"],
    )
    for a in angles_to_set:
        if a in required_dihedrals:
            continue
        assert a in dists_and_angles
        if a == "tau" or a == "N:CA:C":
            nerf_build_kwargs["bond_angle_ca_c"] = dists_and_angles[a]
        elif a == "CA:C:1N":
            nerf_build_kwargs["bond_angle_c_n"] = dists_and_angles[a]
        elif a == "C:1N:1CA":
            nerf_build_kwargs["bond_angle_n_ca"] = dists_and_angles[a]
        else:
            raise ValueError(f"Unrecognized angle: {a}")

    for d in dists_to_set:
        assert d in dists_and_angles.columns
        if d == "0C:1N":
            nerf_build_kwargs["bond_len_c_n"] = dists_and_angles[d]
        elif d == "N:CA":
            nerf_build_kwargs["bond_len_n_ca"] = dists_and_angles[d]
        elif d == "CA:C":
            nerf_build_kwargs["bond_len_ca_c"] = dists_and_angles[d]
        else:
            raise ValueError(f"Unrecognized distance: {d}")

    nerf_builder = nerf.NERFBuilder(**nerf_build_kwargs)
    coords = (
        nerf_builder.centered_cartesian_coords
        if center_coords
        else nerf_builder.cartesian_coords
    )
    coords=nerf_builder.centered_cartesian_coords
    if np.any(np.isnan(coords)):
        logging.warning(f"Found NaN values, not writing pdb file {out_fname}")
        return ""

    assert coords.shape == (
        int(dists_and_angles.shape[0] * 3),
        3,
    ), f"Unexpected shape: {coords.shape} for input of {len(dists_and_angles)}"
    return write_coords_to_pdb(coords, out_fname)


def write_coords_to_pdb(coords: np.ndarray, out_fname: str) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    assert len(coords) % 3 == 0, f"Expected 3N coords, got {len(coords)}"
    atoms = []
    for i, (n_coord, ca_coord, c_coord) in enumerate(
        (coords[j : j + 3] for j in range(0, len(coords), 3))
    ):
        atom1 = struc.Atom(
            n_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 1,
            res_name="GLY",
            atom_name="N",
            element="N",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom2 = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 2,
            res_name="GLY",
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom3 = struc.Atom(
            c_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 3 + 3,
            res_name="GLY",
            atom_name="C",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atoms.extend([atom1, atom2, atom3])
    full_structure = struc.array(atoms)

    # Add bonds
    full_structure.bonds = struc.BondList(full_structure.array_length())
    indices = list(range(full_structure.array_length()))
    for a, b in zip(indices[:-1], indices[1:]):
        full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)

    # Annotate secondary structure using CA coordinates
    # https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
    # https://academic.oup.com/bioinformatics/article/13/3/291/423201
    # a = alpha helix, b = beta sheet, c = coil
    # ss = struc.annotate_sse(full_structure, "A")
    # full_structure.set_annotation("secondary_structure_psea", ss)

    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)
    return out_fname
'''
original_coords,my_data=extract_backbone('./data/107M_A.pdb')
write_coords_to_pdb(original_coords,"original.pdb")
create_new_chain_nerf("recovery.pdb",my_data)
'''