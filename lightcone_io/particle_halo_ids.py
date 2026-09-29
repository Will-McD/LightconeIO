#!/bin/env python

import os
import sys
import argparse
import numpy as np
import time
t0 = time.time()
import psutil
import datetime as dt
import h5py
import scipy.spatial
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

import virgo.util.match as match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
import virgo.mpi.util as mpi_util


# Constants to identify methods for dealing with particles in multiple halos
FRACTIONAL_RADIUS=0
MOST_MASSIVE=1
LEAST_MASSIVE=2
MASS_WEIGHTED=3 # assign overlapping particles by mass
overlap_methods = {
    "fractional-radius" : FRACTIONAL_RADIUS,
    "most-massive"      : MOST_MASSIVE,
    "least-massive"     : LEAST_MASSIVE,
    "mass-weighted"     : MASS_WEIGHTED,
    }

KNOWN_BOX_RESOLUTIONS = ("L1000N1800", "L1000N0900", "L1000N3600", "L2800N5040")
KNOWN_PARTICLE_TYPES = ("BH", "DM", "Gas", "Neutrino", "Stars")

def detect_box_resolution(path):
    """
    """
    return next((part for part in os.path.normpath(path).split(os.sep) if part in KNOWN_BOX_RESOLUTIONS), None)

def batched(seq, batch_size):
    """
    Yield successive batch_size-sized chunks of seq. batch_size=None yields
    the whole seq as a single chunk (no batching). 
    """
    if batch_size is None:
        yield seq
        return
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]

def message(m):
    if comm_rank == 0:
        t1 = time.time()
        elapsed = t1-t0
        print(f"{elapsed:.1f}s: {m}")

def rank_message(m, rank):
    """
    Print a new message on each rank.
    No rank barrier as we want independant messages 
    from each rank with time of arrival shown.
    """
    
    current_time=dt.datetime.now()
    time_str=current_time.strftime("%H:%M:%S")
    print('\t[Rank {rank_nr:03d}] [@{print_time}]'.format(rank_nr=rank,print_time=time_str) + m)

def attr_scalar(value):
    """
    Some HDF5 attributes that are conceptually scalar are stored with a
    length-1 array. Correct by only returning single element. 
    """
    return np.asarray(value).flat[0]


def snapshot_number_redshifts(boxsize_resolution, snapshot_number, inverse=False):
    """
    Returns the redshift of snapshot number.
        If inverse = True, returns snapshot number for redshift passed as snapshot_number param
    """
    if (boxsize_resolution=="L1000N1800") or (boxsize_resolution=="L1000N0900"):
        snapshot_numbers=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77])
        redshift=np.array([15, 10.38, 9.51, 8.7, 7.95, 7.26, 6.63, 6.04, 5.5, 5, 4.75, 4.5, 4.25, 4, 3.75, 3.5, 3.25, 3, 2.95, 2.9, 2.85, 2.8, 2.75, 2.7, 2.65, 2.6, 2.55, 2.5, 2.45, 2.4, 2.35, 2.3, 2.25, 2.2, 2.15, 2.1, 2.05, 2, 1.95, 1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.])
    elif (boxsize_resolution=="L1000N3600") or (boxsize_resolution=="L2800N5040"):
        snapshot_numbers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30, 31, 32, 33, 34, 35, 36, 37, 38, 39,40, 41, 42, 43, 44, 45, 46, 47, 48, 49,50, 51, 52, 53, 54, 55, 56, 57, 58, 59,60, 61, 62, 63, 64, 65, 66, 67, 68, 69,70, 71, 72, 73, 74, 75, 76, 77, 78])
        redshift = np.array([15, 12.26, 10.38, 9.51, 8.7, 7.95, 7.26, 6.63, 6.04, 5.5,5, 4.75, 4.5, 4.25, 4, 3.75, 3.5, 3.25, 3, 2.95,2.9, 2.85, 2.8, 2.75, 2.7, 2.65, 2.6, 2.55, 2.5, 2.45,2.4, 2.35, 2.3, 2.25, 2.2, 2.15, 2.1, 2.05, 2, 1.95,1.9, 1.85, 1.8, 1.75, 1.7, 1.65, 1.6, 1.55, 1.5, 1.45,1.4, 1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1, 0.95,0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.])
    else:
        raise ValueError("boxsize_resolution not recognised")

    if inverse:
        idx = np.argmin(np.abs(redshift - snapshot_number))
        if not np.isclose(redshift[idx], snapshot_number, atol=1e-6):
            raise ValueError(f"redshift {snapshot_number} not found for {boxsize_resolution} (nearest available: {redshift[idx]})")
        return int(snapshot_numbers[idx])

    return snapshot_numbers, redshift


def _fallback_snapshot_table(box_resolution):
    """
    Build (first_snap, last_snap, snapshot_numbers, minimum_redshifts,
    maximum_redshifts) from snapshot_number_redshifts() when a halo lightcone 
    has no Snapshots group at all. 
    Some simulations lack this metadata even though the halo data itself
    is present and populated.
    """
    snapshot_numbers, z = snapshot_number_redshifts(box_resolution, None)
    z = np.asarray(z, dtype=float)
    n = len(z)
    minimum_redshifts = np.empty(n)
    maximum_redshifts = np.empty(n)
    for i in range(n):
        # z decreases as snapshot number/index increases (see the table above)
        maximum_redshifts[i] = z[i] if i == 0 else 0.5 * (z[i] + z[i - 1])
        minimum_redshifts[i] = 0.0 if i == n - 1 else 0.5 * (z[i] + z[i + 1])

    return int(snapshot_numbers.min()), int(snapshot_numbers.max()), snapshot_numbers, minimum_redshifts, maximum_redshifts


def _rss_used():
    """
    Report this process's own resident set size (RSS)
    """
    # Do nothing if psutil is not installed
    if psutil is None:
        return None

    GB = 1024**3
    return psutil.Process().memory_info().rss / GB

def report_rss(m, comm):
    """Collective: report max and total peak RSS across all ranks."""
    rss_gb = _rss_used()
    max_rss = comm.allreduce(rss_gb, op=MPI.MAX)
    sum_rss = comm.allreduce(rss_gb, op=MPI.SUM)
    message(f"RSS [{m}]: max = {max_rss:.2f} [GB], sum = {sum_rss:.2f} [GB]")


def get_halo_lightcone_file_idx(halo_lightcone_filenames, comm, zmin=None, zmax=None, margin_snapshots=1, box_resolution=None):
    """
    hbt halo lightcones do not consistently have the "Header/NumberOfFiles" attribute for phdf5.MultiFile. 
    Instead, use the fact that each file instead covers one snapshot to get file_idx array for phdf5.MultiFile.

    Params:
        :(str) halo_lightcone_filenames :    string for the format of the halo lightcone filename with {file_nr} as a variable. 
                                                'path/to/halo/lightcone/files/file_for_snapshot_{file_nr}.hdf5'
        :(float) zmin, zmax             :   if None use full range of halo lightcone. If given restrict the result to snapshots whose shell
                                                [minimum_redshifts, maximum_redshifts) overlaps with [zmin, zmax], padded by
                                                margin_snapshots extra snapshots on each side.
        :(int) margin_snapshots         :   extra snapshots added as a buffer on each side of zmin and zmax. 
        :(str) box_resolution           :   box size and resolution identifier for simulation, e.g. 'L2800N5040'. Required for when 
                                                "Header/NumberOfFiles" attribute is not found so that the hard coded snapshot number 
                                                redshifts can be used as a fall back. 

    Returns:
        array of file (i.e. snapshot) numbers for phdf5.MultiFile
    """
    comm_rank = comm.Get_rank()
    need_redshifts = (zmin is not None) or (zmax is not None)
    if comm_rank == 0:
        filename0 = halo_lightcone_filenames % {"file_nr": 0}
        with h5py.File(filename0, "r") as infile:
            has_snapshots_group = "Snapshots" in infile
            if has_snapshots_group:
                snaps = infile["Snapshots"]
                first_snap = int(attr_scalar(snaps.attrs["first_snapshot_number"]))
                last_snap = int(attr_scalar(snaps.attrs["last_snapshot_number"]))
                if need_redshifts:
                    snapshot_numbers = np.asarray(snaps.attrs["snapshot_numbers"])
                    minimum_redshifts = np.asarray(snaps.attrs["minimum_redshifts"])
                    maximum_redshifts = np.asarray(snaps.attrs["maximum_redshifts"])
                else:
                    snapshot_numbers = None
                    minimum_redshifts = None
                    maximum_redshifts = None
            else:
                first_snap = None
                last_snap = None
                snapshot_numbers = None
                minimum_redshifts = None
                maximum_redshifts = None
    else:
        has_snapshots_group = None
        first_snap = None
        last_snap = None
        snapshot_numbers = None
        minimum_redshifts = None
        maximum_redshifts = None

    (has_snapshots_group, first_snap, last_snap, snapshot_numbers, minimum_redshifts, maximum_redshifts) = comm.bcast((has_snapshots_group, first_snap, last_snap, snapshot_numbers, minimum_redshifts, maximum_redshifts))
    
    if not has_snapshots_group:
        message(f"Halo lightcone catalogue has no Snapshots group. Using the hardcoded snapshot-redshift range table for box_resolution={box_resolution} instead")
        if box_resolution is None:
            raise RuntimeError("box-resolution must be given so the hardcoded snapshot-redshift range table can be used instead")
        
        # Pure computation from the hardcoded table, every rank independently does this, no need to broadcast 
        first_snap, last_snap, snapshot_numbers, minimum_redshifts, maximum_redshifts = _fallback_snapshot_table(box_resolution)

    if not need_redshifts:
        return np.arange(first_snap, last_snap + 1)

    overlap = (minimum_redshifts <= zmax) & (maximum_redshifts >= zmin)
    matched_idx = np.nonzero(overlap)[0]
    if len(matched_idx) == 0:
        raise RuntimeError(f"No halo lightcone snapshots overlap redshift range [{zmin}, {zmax}]")
    i0 = max(matched_idx[0] - margin_snapshots, 0)
    i1 = min(matched_idx[-1] + margin_snapshots, len(snapshot_numbers) - 1)
    return snapshot_numbers[i0:i1 + 1]


def get_snapshot_scale_factors(halo_lightcone_filenames, comm, box_resolution=None):
    """
    Returns the scalefactor corresponding to the snapshots of the halo lightcone. 
    Reads the redshift of every snapshot in lightcone/Snapshots.
    
    Params:
        :(str) halo_lightcone_filenames :    string for the format of the halo lightcone filename with {file_nr} as a variable. 
                                                    'path/to/halo/lightcone/files/file_for_snapshot_{file_nr}.hdf5'
        :(str) box_resolution           :   box size and resolution identifier for simulation, e.g. 'L2800N5040'. Required for when 
                                                    "Header/NumberOfFiles" attribute is not found so that the hard coded snapshot number 
                                                    redshifts can be used as a fall back. 
    Returns
        {snapshot numbers: snapshot scalefactors}
    """
    comm_rank = comm.Get_rank()
    if comm_rank == 0:
        filename0 = halo_lightcone_filenames % {"file_nr": 77}
        with h5py.File(filename0, "r") as infile:
            has_snapshots_group = "Snapshots" in infile
            if has_snapshots_group:

                snaps = infile["Snapshots"]
                snapshot_numbers = np.asarray(snaps.attrs["snapshot_numbers"])
                snapshot_redshifts = np.asarray(snaps.attrs["snapshot_redshifts"])
                snapshot_expansion_factors=1.0/(1.0+snapshot_redshifts.astype(float))

            else:
                snapshot_numbers = None
                snapshot_expansion_factors = None
    else:
        has_snapshots_group = None
        snapshot_numbers = None
        snapshot_expansion_factors=None
    
    has_snapshots_group, snapshot_numbers, snapshot_expansion_factors = comm.bcast((has_snapshots_group, snapshot_numbers, snapshot_expansion_factors))
    
    if not has_snapshots_group:
        if box_resolution is None:
            raise RuntimeError(
                "Halo lightcone catalogue has no Snapshots group; --box-resolution must be "
                "given so the hardcoded snapshot/redshift table can be used instead")
        
        snapshot_numbers, snapshot_redshifts = snapshot_number_redshifts(box_resolution, None)
        snapshot_expansion_factors=1.0/(1.0+snapshot_redshifts.astype(float))

    return {int(sn) : sz for sn, sz in zip(snapshot_numbers, snapshot_expansion_factors)}

def read_lightcone_halo_positions_and_radii(args, radius_name, mass_name, zmin=None, zmax=None, margin_snapshots=1):
    """
    Read in the lightcone halo catalogue and cross reference with SOAP
    to find the SO radius and mass for each halo in the lightcone.

    Assumes that positions in the lightcone are comoving and in the
    same units as SOAP (except for the expansion factor dependence).
    
    If zmin/zmax are given, only halo lightcone snapshots overlapping that
    redshift range (plus margin_snapshots of padding) are read. 

    """

    # Parallel read the halo catalogue: need (x,y,z), snapnum, id
    message("Reading lightcone halo catalogue")

    # Updated to read in halo lightcone as they are currently constructed.
    # Changed the names of the properties, e.g., "Pos_minpot" -> "Lightcone/HaloCentre",  "SnapNum" -> "SnapshotNumber", ect...
    # Halo coordinates in the lightcone are now an nx3 array as opposed to 3 1D arrays. 

    halo_lightcone_datasets = ("Lightcone/HaloCentre", "Lightcone/SnapshotNumber", "InputHalos/HaloCatalogueIndex","InputHalos/SOAPIndex") 

    file_idx = get_halo_lightcone_file_idx(args.halo_lightcone_filenames, comm, zmin=zmin, zmax=zmax, margin_snapshots=margin_snapshots,box_resolution=args.box_resolution)
    mf = phdf5.MultiFile(args.halo_lightcone_filenames, file_idx=file_idx, comm=comm)

    halo_lightcone_data = mf.read(halo_lightcone_datasets, group="/", read_attributes=True)

    # Store index in halo lightcone of each halo
    nr_local_halos = len(halo_lightcone_data["InputHalos/HaloCatalogueIndex"]) # update property name "ID" -> "InputHalos/HaloCatalogueIndex"
    offset = comm.scan(nr_local_halos) - nr_local_halos
    halo_lightcone_data["IndexInHaloLightcone"] = np.arange(nr_local_halos, dtype=int) + offset

    # Repartition halos for better load balancing
    message("Repartition halo catalogue")
    nr_local_halos = len(halo_lightcone_data["InputHalos/HaloCatalogueIndex"]) # update property name
    nr_total_halos = comm.allreduce(nr_local_halos)
    nr_desired = np.zeros(comm_size, dtype=int)
    nr_desired[:] = nr_total_halos // comm_size
    nr_desired[:nr_total_halos % comm_size] += 1
    assert np.sum(nr_desired) == nr_total_halos
    for name in halo_lightcone_data:
        halo_lightcone_data[name] = psort.repartition(halo_lightcone_data[name], nr_desired, comm=comm)

    # The input catalogue is ordered by redshift, but we want a mix of redshifts on each rank
    message("Reassign halos to MPI ranks")
    nr_local_halos = len(halo_lightcone_data["InputHalos/HaloCatalogueIndex"])
    rng = np.random.default_rng()
    sort_key = rng.integers(comm_size, size=nr_local_halos, dtype=np.int32)
    order = psort.parallel_sort(sort_key, comm=comm, return_index=True)
    for name in sorted(halo_lightcone_data):
        psort.fetch_elements(halo_lightcone_data[name], order, result=halo_lightcone_data[name], comm=comm)

    # Sort locally by snapnum
    message("Sorting local lightcone halos by snapshot")
    #order = np.argsort(halo_lightcone_data["SnapNum"])
    order = np.argsort(halo_lightcone_data["Lightcone/SnapshotNumber"]) # update property name
    for name in halo_lightcone_data:
        halo_lightcone_data[name] = halo_lightcone_data[name][order,...]

    # Find range of local halos at each snapshot
    message("Identifying halos at each snapshot")
    unique_snap, snap_offset, snap_count = np.unique(halo_lightcone_data["Lightcone/SnapshotNumber"],
                                                     return_index=True, return_counts=True)

    # Find full range of snapshots across all MPI ranks
    # A rank may have no local halos at all, in which case np.amin/np.amax would raise on the empty unique_snap array.
    if len(unique_snap) > 0:
        local_min_snap = np.amin(unique_snap)
        local_max_snap = np.amax(unique_snap)
    else:
        local_min_snap = np.iinfo(unique_snap.dtype).max
        local_max_snap = -1
    min_snap = comm.allreduce(local_min_snap, op=MPI.MIN)
    max_snap = comm.allreduce(local_max_snap, op=MPI.MAX)

    # Make snapnum, count and offset arrays which include snapshots not present on this rank:
    # We're going to do collective reads of the SOAP outputs so all ranks need to agree on
    # what range of snapshots to do.
    nr_snaps = max_snap - min_snap + 1
    unique_snap_all = np.arange(min_snap, max_snap+1, dtype=int)
    snap_offset_all = np.zeros(nr_snaps, dtype=int)
    snap_count_all  = np.zeros(nr_snaps, dtype=int)
    for us, so, sc in zip(unique_snap, snap_offset, snap_count):
        i = us - min_snap
        assert unique_snap_all[i] == us
        snap_offset_all[i] = so
        snap_count_all[i] = sc

    # Allocate storage for radius of each lightcone halo
    nr_halos = len(halo_lightcone_data["InputHalos/HaloCatalogueIndex"])
    halo_lightcone_data[radius_name] = None # Don't know dtype for radius array yet

    # Get scalefactors of all snapshots 
    scale_factor_of_snapshot = get_snapshot_scale_factors(args.halo_lightcone_filenames, comm, box_resolution=args.box_resolution)

    # Loop over snapshots
    for snapnum in unique_snap_all:

        # Datasets to read from SOAP
        soap_datasets = ("InputHalos/HaloCatalogueIndex", radius_name, mass_name) # Updated to work with HBT-SOAP. 

        # Read the SOAP catalogue for this snapshot
        message(f"Reading SOAP output for snapshot {snapnum}")
        mf = phdf5.MultiFile(args.soap_filenames % {"snap_nr" : snapnum}, file_idx=(0,), comm=comm)
        soap_data = mf.read(soap_datasets, read_attributes=True)

        # Get the expansion factor of this snapshot
        a = scale_factor_of_snapshot[snapnum]

        # Ensure radii are in comoving units
        radius_a_exponent = float(attr_scalar(soap_data[radius_name].attrs["a-scale exponent"]))
        soap_data[radius_name] *= a**(radius_a_exponent-1.0)

        # Match lightcone halos at this snapshot to SOAP halos by ID
        message("Finding lightcone halos in SOAP output")
        i1 = snap_offset_all[snapnum-min_snap]
        i2 = snap_offset_all[snapnum-min_snap] + snap_count_all[snapnum-min_snap]
        assert np.all(halo_lightcone_data["Lightcone/SnapshotNumber"][i1:i2] == snapnum)

        # Matching and Sorting step complete by "InputHalos/SOAPIndex", so no need ot repeat now 
        ptr = halo_lightcone_data["InputHalos/SOAPIndex"][i1:i2]
        assert np.all(ptr>=0) # All halos in the lightcone should be found in SOAP

        # Allocate storage for radii now that we know what dtype SOAP uses
        if halo_lightcone_data[radius_name] is None:
            radius_dtype = soap_data[radius_name].dtype
            halo_lightcone_data[radius_name] = phdf5.AttributeArray(-np.ones(nr_halos, dtype=radius_dtype),
                                                                    attrs=soap_data[radius_name].attrs)
            mass_dtype = soap_data[mass_name].dtype
            halo_lightcone_data[mass_name] = phdf5.AttributeArray(-np.ones(nr_halos, dtype=mass_dtype),
                                                                  attrs=soap_data[mass_name].attrs)

        message("Storing SO radii for lightcone halos at this snapshot")
        psort.fetch_elements(soap_data[radius_name], ptr,
                             result=halo_lightcone_data[radius_name][i1:i2], comm=comm)

        message("Storing SO masses for lightcone halos at this snapshot")
        psort.fetch_elements(soap_data[mass_name], ptr,
                             result=halo_lightcone_data[mass_name][i1:i2], comm=comm)

    # All halos should have been assigned a radius
    assert np.all(halo_lightcone_data[radius_name] >= 0)
    assert np.all(halo_lightcone_data[mass_name] >= 0)

    return halo_lightcone_data


def read_lightcone_index(args):
    """
    Read the index file and determine names of all particle files and which particle types are present
    """
    
    # Particle types which may be in the lightcone:("BH", "DM", "Gas", "Neutrino", "Stars") 
    type_names = tuple(args.particle_types)
    type_z_range = {}

    # Now, find the lightcone particle output and read the index info
    index_file = args.lightcone_dir+"/"+args.lightcone_base+"_index.hdf5"
    if comm_rank == 0:
        with h5py.File(index_file, "r") as index:
            lc = index["Lightcone"]
            nr_mpi_ranks = int(attr_scalar(lc.attrs["nr_mpi_ranks"]))
            final_file_on_rank = lc.attrs["final_particle_file_on_rank"]
            for tn in type_names:
                min_z = float(attr_scalar(lc.attrs["minimum_redshift_"+tn]))
                max_z = float(attr_scalar(lc.attrs["maximum_redshift_"+tn]))
                if max_z > min_z:
                    type_z_range[tn] = (min_z, max_z)
    else:
        nr_mpi_ranks = None
        final_file_on_rank = None
        type_z_range = None
    nr_mpi_ranks, final_file_on_rank, type_z_range = comm.bcast((nr_mpi_ranks, final_file_on_rank, type_z_range))

    # Report which particle types we found
    for name in type_z_range:
        min_z, max_z = type_z_range[name]
        message(f"have particles for type {name} from z={min_z} to z={max_z}")

    # Make a full list of files to read
    all_particle_files = []
    for rank_nr in range(nr_mpi_ranks):
        for file_nr in range(final_file_on_rank[rank_nr]+1):
            filename = f"{args.lightcone_dir}/{args.lightcone_base}_particles/{args.lightcone_base}_{file_nr:04d}.{rank_nr}.hdf5"
            all_particle_files.append(filename)
        
    return type_z_range, all_particle_files


def compute_particle_group_index(halo_id, halo_pos, halo_radius, halo_mass, part_pos,
                                 overlap_method):
    """
    Tag particles which are within the specified radius of a halo
    """

    # Assign indexes to the particles so we can restore their ordering later
    nr_particles = part_pos.shape[0]
    offset = comm.scan(nr_particles) - nr_particles
    part_index = np.arange(nr_particles, dtype=np.int64) + offset

    nr_halos = halo_pos.shape[0]
    nr_halos_total = comm.allreduce(nr_halos)
    nr_particles_total = comm.allreduce(nr_particles)
    message(f"Have {nr_particles_total} particles and {nr_halos_total} halos")

    # Will split the halos and particles by x coordinate, with a roughly
    # constant number of particles per rank. First, sort the particles by x.
    message("Sorting particles by x coordinate")
    sort_key = part_pos[:,0].copy()
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    del sort_key
    psort.fetch_elements(part_pos, order, result=part_pos, comm=comm)
    psort.fetch_elements(part_index, order, result=part_index, comm=comm)
    del order

    # Find the maximum halo radius. 
    local_max_radius = np.amax(halo_radius) if len(halo_radius) > 0 else 0.0 # Ranks with 0 haloes create an empty array and now np.amax will return 0.0 instead of raising an error
    max_radius = comm.allreduce(local_max_radius, op=MPI.MAX)

    # Find maximum distance to any particle.
    if part_pos.shape[0] > 0:
        local_max_particle_distance = np.amax(np.sqrt(np.sum(part_pos**2, axis=1)))
    else:
        local_max_particle_distance = 0.0 # Guard against ranks with no local particles causing np.amax to raise an error on an empty array.
    max_particle_distance = comm.allreduce(local_max_particle_distance, op=MPI.MAX)

    # Find the subset of halos which can overlap the particle distribution:
    # This helps in case the halo lightcone goes out to much higher redshift
    # than the particle lightcone.
    halo_distance = np.sqrt(np.sum(halo_pos**2, axis=1))
    within_distance = halo_distance < (max_particle_distance + max_radius)
    halo_pos = halo_pos[within_distance,:]
    halo_id = halo_id[within_distance]
    halo_radius = halo_radius[within_distance]
    halo_mass = halo_mass[within_distance]
    nr_halos_left = comm.allreduce(halo_id.shape[0], op=MPI.SUM)
    message(f"Halos within redshift range = {nr_halos_left} of {nr_halos_total}")

    # Determine the range of x coordinates of halos which could overlap particles on this rank.
    # A rank with no local particles reports a degenerate range (+inf) instead of calling np.amin/np.amax on an empty array. 
    # searchsorted() correctly selects zero haloes for ranks with no local particles rather than crashing or computing a negative count.
    if part_pos.shape[0] > 0:
        local_x_min = np.amin(part_pos[:,0]) - max_radius
        local_x_max = np.amax(part_pos[:,0]) + max_radius
    else:
        local_x_min = np.inf
        local_x_max = np.inf
    x_min_on_rank = np.asarray(comm.allgather(local_x_min), dtype=part_pos.dtype)
    x_max_on_rank = np.asarray(comm.allgather(local_x_max), dtype=part_pos.dtype)

    # Sort local halos by x coordinate
    report_rss("Sorting local lightcone halos by x coordinate", comm) # message and update on rank mem being used
    order = np.argsort(halo_pos[:,0])
    halo_pos = halo_pos[order,:]
    halo_id = halo_id[order]
    halo_radius = halo_radius[order]
    halo_mass = halo_mass[order]
    del order

    # Determine what range of halos needs to be sent to each MPI rank
    first_halo_for_rank = np.searchsorted(halo_pos[:,0], x_min_on_rank, side="left")
    last_halo_for_rank = np.searchsorted(halo_pos[:,0], x_max_on_rank, side="right")
    nr_halos_for_rank = last_halo_for_rank - first_halo_for_rank
    nr_halos_for_rank_total = comm.allreduce(nr_halos_for_rank)
    total_nr_halos_read = comm.allreduce(halo_pos.shape[0])
    total_nr_halos_sent = np.sum(nr_halos_for_rank_total)
    duplication_factor = total_nr_halos_sent / total_nr_halos_read
    message(f"Minimum halos on rank after exchange = {np.amin(nr_halos_for_rank_total)}")
    message(f"Maximum halos on rank after exchange = {np.amax(nr_halos_for_rank_total)}")
    message(f"Duplication factor = {duplication_factor}")
    
    report_rss("With duplications", comm) # message and update on rank mem being used

    # Compute lengths and offsets for alltoallv halo exchange
    send_offset = first_halo_for_rank
    send_count = nr_halos_for_rank
    recv_count = np.asarray(comm.alltoall(send_count), dtype=send_count.dtype)
    recv_offset = np.cumsum(recv_count) - recv_count

    # Exchange halo IDs
    message("Exchanging halo IDs")
    halo_id_recv = np.empty_like(halo_id, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_id, send_count, send_offset,
                       halo_id_recv, recv_count, recv_offset,
                       comm=comm)
    halo_id = halo_id_recv
    del halo_id_recv
    comm.barrier()

    # Exchange halo radii
    message("Exchanging halo radii")
    halo_radius_recv = np.empty_like(halo_radius, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_radius, send_count, send_offset,
                       halo_radius_recv, recv_count, recv_offset,
                       comm=comm)
    halo_radius = halo_radius_recv
    del halo_radius_recv
    comm.barrier()

    # Exchange halo masses
    message("Exchanging halo masses")
    halo_mass_recv = np.empty_like(halo_mass, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_mass, send_count, send_offset,
                       halo_mass_recv, recv_count, recv_offset,
                       comm=comm)
    halo_mass = halo_mass_recv
    del halo_mass_recv
    comm.barrier()
    
    # Exchange halo positions:
    # These are vectors so flatten, exchange then restore shape
    message("Exchanging halo positions")
    halo_pos.shape = (-1,)
    halo_pos_recv = np.empty_like(halo_pos, shape=3*np.sum(recv_count))
    psort.my_alltoallv(halo_pos, send_count*3, send_offset*3,
                       halo_pos_recv, recv_count*3, recv_offset*3,
                       comm=comm)
    halo_pos = halo_pos_recv
    halo_pos.shape = (-1, 3)
    del halo_pos_recv
    comm.barrier()

    # Build a kdtree with the local particles
    message("Building kdtree")
    tree = scipy.spatial.KDTree(part_pos)

    # Allocate output array for the particle halo IDs etc
    nr_parts = part_pos.shape[0]
    part_halo_id = -np.ones(nr_parts, dtype=np.int64)       # ID of halo particle is assigned to
    part_halo_mass = np.ndarray(nr_parts, dtype=np.float32) # Mass of the halo
    if overlap_method == LEAST_MASSIVE:
        # Looking for least massive halo, so initialize mass to huge value
        part_halo_mass[:] = np.finfo(part_halo_mass.dtype).max
    else:
        # Looking for most massive halo or not using mass, so initialize mass to -1
        part_halo_mass[:] = -1
    part_halo_r_frac_2 = -np.ndarray(nr_parts, dtype=np.float32) # Smallest ((Particle radius)/(halo r200))**2 so far
    part_halo_r_frac_2[:] = np.finfo(part_halo_r_frac_2.dtype).max  # Initialize min. fractional radius to huge value

    # Report maximum halo radius. As above, this rank may have received no halos
    # through the exchange, so guard against np.amax on an empty array.
    local_max_radius = np.amax(halo_radius) if len(halo_radius) > 0 else 0.0
    max_radius = comm.allreduce(local_max_radius, op=MPI.MAX)
    message(f"Maximum halo radius = {max_radius}")

    # Loop over local halos
    nr_assigned = 0
    report_rss("Assigning halo IDs to particles", comm) # message and report rank mem use
    for i in range(len(halo_id)):
            
        # Identify particles within this halo's radius
        idx = np.asarray(tree.query_ball_point(halo_pos[i,:], halo_radius[i]), dtype=int)

        # Compute radius squared for each particle
        r_part_2 = np.sum((part_pos[idx,:] - halo_pos[i,:])**2.0, axis=1)
        
        # Compute ((particle radius)/(halo radius))**2
        r_frac_2 = r_part_2 / (halo_radius[i]**2) 

        # Identify particles to update
        if overlap_method == FRACTIONAL_RADIUS:
            # Assign particles to this halo if (particle radius)/(halo radius) is smaller
            # than the smallest value so far
            to_update = (r_frac_2 < part_halo_r_frac_2[idx])
        elif overlap_method == MOST_MASSIVE:
            # Assign particles to this halo if this is the most massive halo the particle
            # has been found to be in so far
            to_update = (halo_mass[i] > part_halo_mass[idx])
        elif overlap_method == LEAST_MASSIVE:
            # Assign particles to this halo if this is the least massive halo the particle
            # has been found to be in so far
            to_update = (halo_mass[i] < part_halo_mass[idx])
        elif overlap_method == MASS_WEIGHTED:
            # Assign particles to this halo if (particle radius / halo mass)**2 is smaller
            # than the smallest value so far. 
            # Reuse part_halo_r_frac_2 as scratch storage
            # for this squared metric instead of the true fractional radius, since
            # overlap_method is fixed for the whole run so the two never coexist.
            mass_weighted_metric_2 = r_part_2 / (halo_mass[i]**2)
            to_update = (mass_weighted_metric_2 < part_halo_r_frac_2[idx])
        else:
            raise ValueError("Unrecognized value of overlap_method")        
        
        idx = idx[to_update] # update idx 

        # Tag particles to update with the halo ID, mass and fractional radius
        part_halo_id[idx]       = halo_id[i]
        part_halo_mass[idx]     = halo_mass[i]
        if overlap_method == MASS_WEIGHTED: # if mass weighted upate with mass weighted metric, 
            part_halo_r_frac_2[idx] = mass_weighted_metric_2[to_update]
        else:
            part_halo_r_frac_2[idx] = r_frac_2[to_update] # if not mass weighted upate with fractional metric, 
        nr_assigned            += len(idx)

    nr_assigned_tot = comm.allreduce(nr_assigned)
    fraction_assigned = nr_assigned_tot / nr_particles_total
    message(f"Total particles assigned to halos = {nr_assigned_tot}")
    message(f"Fraction assigned = {fraction_assigned} (inc. duplicates due to halo overlap)")
    report_rss("Assigned haloes to particles in batch", comm)
    # Tidy up
    del halo_id
    del halo_pos
    del halo_radius
    del halo_mass
    del part_pos

    # Return r_frac=r/r200c or r/M200c for particles in halos and -1 for those not in halos
    in_halo = (part_halo_id >= 0)
    part_halo_r_frac = np.where(in_halo, np.sqrt(part_halo_r_frac_2), -1.0)
    del part_halo_r_frac_2

    # Replace any huge halo masses (i.e. particles not in any halo) with -1
    part_halo_mass[in_halo==False] = -1.0

    # Restore original particle ordering and return halo IDs etc
    report_rss("Restoring particle order", comm)
    order = psort.parallel_sort(part_index, return_index=True, comm=comm)
    del part_index
    psort.fetch_elements(part_halo_id, order, result=part_halo_id, comm=comm)
    psort.fetch_elements(part_halo_mass, order, result=part_halo_mass, comm=comm)
    psort.fetch_elements(part_halo_r_frac, order, result=part_halo_r_frac, comm=comm)

    return part_halo_id, part_halo_mass, part_halo_r_frac


def main(args):

    # Determine method to deal with overlapping halos
    overlap_method = overlap_methods[args.overlap_method]
    message(f"Halo overlap method: {args.overlap_method}")

    # Read in position and radius for halos in the lightcone.
    # if args.centrals-only (default):
    #       use SO values, defined for central subhalos only.
    # elif args.all-subhalos:
    #      use BoundSubhalo values, which are defined for centrals and satellites alike.
    if args.centrals_only:
        message(f"Halo radius definition: {args.soap_so_name} (centrals only)")
        radius_name = f"{args.soap_so_name}/SORadius"
        mass_name = f"{args.soap_so_name}/TotalMass"
    else:
        message("Halo radius definition: BoundSubhalo (all subhalos)")
        radius_name = "BoundSubhalo/EnclosedRadius"
        mass_name = "BoundSubhalo/TotalMass"
    
    # Locate the particle data
    type_z_range, all_particle_files = read_lightcone_index(args)
    
    # Instead of reading every snapshot in the halo catalogue, 
    # just read the snapshots that can contain a halo overlapping the redshift 
    # range of any selected particle type. The same combined range is used again
    # below for the pass-through read, so both reads see the same file_idx and
    # IndexInHaloLightcone stays consistent between them.
    combined_zmin = min((type_z_range[t][0] for t in type_z_range), default=None)
    combined_zmax = max((type_z_range[t][1] for t in type_z_range), default=None)
    halo_lightcone_data = read_lightcone_halo_positions_and_radii(
        args, radius_name, mass_name, zmin=combined_zmin, zmax=combined_zmax)

    report_rss("Loaded halo positons and radii", comm) # Message and report rank mem useage

    # Generate filenames for the output:
    # These are the input filenames with the directory replaced with argument output_dir.
    output_filenames = []
    for input_filename in all_particle_files:
        dirname, filename = os.path.split(input_filename)
        output_filenames.append(os.path.join(args.output_dir, filename))

    # Open the input particle file set
    mf = phdf5.MultiFile(all_particle_files, comm=comm)

    # Loop over types to do
    create_files = True
    for ptype in type_z_range:
        
        message(f"\nProcessing particle type: {ptype}")
        mode = "w" if create_files else "r+" # write mode ('w') if first call, otherwise read + update ('r+')
        
        # Determine batches of files. 
        # We reduce the peak overhead cost by update files in batches instead of all files at once. 
        nr_batches = -(-len(all_particle_files) // args.particle_files_per_batch) if args.particle_files_per_batch else 1
        file_batch_iter = zip(batched(all_particle_files, args.particle_files_per_batch),batched(output_filenames, args.particle_files_per_batch))
        for batch_nr, (batch_input_files, batch_output_files) in enumerate(file_batch_iter):
            if nr_batches > 1:
                message(f"Particle file batch {batch_nr + 1}/{nr_batches} for type {ptype}")
        
            # Open files in batches to reduce so peak memory. 
            mf_batch = phdf5.MultiFile(batch_input_files, comm=comm)

            # Read in positions of lightcone particles of this type
            #message(f"Reading particles")
            part_pos = mf_batch.read("Coordinates", group=ptype)
        
            report_rss("set multifile for coordinates", comm)
        
            # Record number of particles read from each file
            elements_per_file = mf_batch.get_elements_per_file("Coordinates", group=ptype)

            # Rebalance particle load between MPI ranks
            nr_parts_per_rank_read = np.asarray(comm.allgather(part_pos.shape[0]), dtype=int)
            nr_parts_total = np.sum(nr_parts_per_rank_read)
            nr_parts_per_rank_balanced = np.zeros_like(nr_parts_per_rank_read)
            nr_av = (nr_parts_total // comm_size)
            nr_parts_per_rank_balanced[:] = nr_av
            nr_parts_per_rank_balanced[:nr_parts_total % comm_size] += 1
            assert np.sum(nr_parts_per_rank_balanced) == nr_parts_total
            part_pos = psort.repartition(part_pos, ndesired=nr_parts_per_rank_balanced, comm=comm)
        
            # Report load balancing
            max_nr_parts = comm.allreduce(part_pos.shape[0], op=MPI.MAX)
            min_nr_parts = comm.allreduce(part_pos.shape[0], op=MPI.MIN)
            message(f"No. of particles per rank min={min_nr_parts}, max={max_nr_parts}")
            report_rss("loaded particles on ranks", comm)

            # Assign group indexes to the particles
            message("Assigning group indexes")
            #report_rss("Assigning group indexes", comm)
            halo_id = halo_lightcone_data["IndexInHaloLightcone"] # we need this. 
            #halo_pos = halo_lightcone_data["Pos_minpot"] 
            halo_pos = halo_lightcone_data["Lightcone/HaloCentre"] # WILL UPDATES: update property name for new halo lightcone format
            halo_radius = halo_lightcone_data[radius_name]
            halo_mass = halo_lightcone_data[mass_name]
            part_halo_id, part_halo_mass, part_halo_r_frac = compute_particle_group_index(halo_id, halo_pos, halo_radius, halo_mass, part_pos, overlap_method)
            del part_pos
            del halo_id
            del halo_pos
            del halo_radius
            del halo_mass
        

            # Restore original partitioning of particles
            part_halo_id = psort.repartition(part_halo_id, ndesired=nr_parts_per_rank_read, comm=comm)
            part_halo_mass = psort.repartition(part_halo_mass, ndesired=nr_parts_per_rank_read, comm=comm)
            part_halo_r_frac = psort.repartition(part_halo_r_frac, ndesired=nr_parts_per_rank_read, comm=comm)
            report_rss("Final repartition", comm)

            # Write the output, appending to file if this is not the first particle type
            message(f"Writing output to {args.output_dir}")

            # determine sorting scheme dset name 
            if overlap_method == MASS_WEIGHTED:
                if mass_name =="BoundSubhalo/TotalMass":
                    metric_column_name="BoundMassWeightedMetric"
                elif (mass_name==f"{args.soap_so_name}/TotalMass") or args.centrals_only==True:
                    metric_column_name="MassWeightedMetric_"+args.soap_so_name.split("/")[-1]
                else:
                    metric_column_name = "MassWeightedMetric"
            else:
                metric_column_name="FractionalRadius"
            
            # determine halo mass dset name 
            if mass_name =="BoundSubhalo/TotalMass":
                HaloMass_column_name="TotalBoundMass"
            elif (mass_name==f"{args.soap_so_name}/TotalMass") or args.centrals_only==True:
                HaloMass_column_name="HaloMass_"+args.soap_so_name.split("/")[-1]
            else:
                HaloMass_column_name="HaloMass"

            datasets = {
                "IndexInHaloLightcone" : part_halo_id,
                metric_column_name : part_halo_r_frac,
                HaloMass_column_name : part_halo_mass,
            }
            attributes = {
                "IndexInHaloLightcone" : halo_lightcone_data["InputHalos/HaloCatalogueIndex"].attrs,
                metric_column_name : halo_lightcone_data["InputHalos/HaloCatalogueIndex"].attrs,
                HaloMass_column_name : halo_lightcone_data[mass_name].attrs,
            }
            mf_batch.write(datasets, elements_per_file, batch_output_files, mode,
                     group=ptype, attrs=attributes, gzip=6, shuffle=True)

            # Tidy up before reading next particle type
            del part_halo_id
            del part_halo_r_frac
            del part_halo_mass

        # Only need to create new output files for the first type
        create_files = False
        
    comm.barrier()

    # Discard reordered halo lightcone data
    del halo_lightcone_data
    del mf_batch

    message("Reading lightcone halo properties to copy to output particle files")

    halo_properties = (
        "InputHalos/HaloCatalogueIndex",
        "Lightcone/SnapshotNumber",
        "InputHalos/SOAPIndex"
    )
    
    # Must use the same zmin/zmax as above, so this read's file_idx lines up with the read used for matching.
    file_idx = get_halo_lightcone_file_idx(args.halo_lightcone_filenames, comm, zmin=combined_zmin, zmax=combined_zmax,box_resolution=args.box_resolution)
    
    mf_in = phdf5.MultiFile(args.halo_lightcone_filenames, file_idx=file_idx, comm=comm)
    halo_lightcone_data = mf_in.read(halo_properties, read_attributes=True)

    halo_lightcone_data["InputHalos/HaloCatalogueIndex"] = halo_lightcone_data["InputHalos/HaloCatalogueIndex"].astype(np.int64) # Avoid using unsigned int

    # Loop over particle types to update
    nr_batches = -(-len(output_filenames) // args.particle_files_per_batch) if args.particle_files_per_batch else 1
    for ptype in type_z_range:
        for batch_nr, batch_output_files in enumerate(batched(output_filenames, args.particle_files_per_batch)):
            if nr_batches > 1:
                message(f"Pass-through file batch {batch_nr + 1}/{nr_batches} for type {ptype}")

            # Open just this batch of output files, for the same reason as the matching loop above.
            mf_out_batch = phdf5.MultiFile(batch_output_files, comm=comm)

            message(f"Reading halo index for particles of type {ptype}")
            halo_index = mf_out_batch.read(f"{ptype}/IndexInHaloLightcone")
            elements_per_file = mf_out_batch.get_elements_per_file(f"{ptype}/IndexInHaloLightcone")
        
            for prop_name in halo_properties:
                message(f"Pass through quantity {prop_name} for type {ptype}")
                in_halo = (halo_index >= 0)
                dtype = halo_lightcone_data[prop_name].dtype
                shape = (halo_index.shape[0],)+halo_lightcone_data[prop_name].shape[1:]
                prop_data = -np.ones(shape, dtype=dtype) # Set property=-1 if particle not in halo
                prop_data[in_halo,...] = psort.fetch_elements(halo_lightcone_data[prop_name], halo_index[in_halo], comm=comm)
                # Write the new dataset to the output files
                dataset_name = f"{ptype}/{prop_name.split('/')[-1]}"
                mf_out_batch.write({dataset_name : prop_data}, elements_per_file, batch_output_files, "r+",
                             attrs={dataset_name : halo_lightcone_data[prop_name].attrs},
                             gzip=6, shuffle=True)
            del halo_index
            del mf_out_batch

    
if __name__ == "__main__":

    # Get command line arguments
    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(description='Create lightcone halo catalogues.', comm=comm)
    parser.add_argument('lightcone_dir',  help='Directory with lightcone particle outputs')
    parser.add_argument('lightcone_base', help='Base name of the lightcone to use')
    parser.add_argument('halo_lightcone_filenames', help='Format string to generate halo lightcone filenames')
    parser.add_argument('soap_filenames', help='Format string to generate SOAP filenames')
    parser.add_argument('output_dir',     help='Where to write the output')
    parser.add_argument('--soap-so-name', type=str, default="SO/200_crit",
                        help='Name of SOAP group with the halo mass and radius, e.g. "SO/200_crit". Only used when --centrals-only is set.')
    parser.add_argument('--centrals-only', dest='centrals_only', action='store_true', default=True,
                        help='Only match particles against central subhalos, using SO values (--soap-so-name) for radius and mass. This is the default.')
    parser.add_argument('--all-subhalos', dest='centrals_only', action='store_false',
                        help='Match particles against all subhalos, both centrals and satellites, using BoundSubhalo/EnclosedRadius and BoundSubhalo/TotalMass for radius and mass instead of SO values.')
    parser.add_argument('--overlap-method', type=str, default="fractional-radius", choices=list(overlap_methods),
                        help="How to assign particles which are in overlapping halos")
    parser.add_argument('--particle-types', type=str, nargs='+', default=list(KNOWN_PARTICLE_TYPES),
                        choices=KNOWN_PARTICLE_TYPES,
                        help='Which particle types to process. Default: all known types. A type is skipped if the particle lightcone has no data for it.')
    parser.add_argument('--particle-files-per-batch', type=int, default=512,
                        help='Process each particle type in batches of this many input/output files at a time, instead of reading the whole type at once. Reduces peak memory at the cost of repeating the halo-exchange/matching setup once per batch instead of once per type. Default: no batching (read/write every file for a type in one go).')

    args = parser.parse_args()
    
    args.box_resolution = detect_box_resolution(args.lightcone_dir)

    message(f"Starting on {comm_size} MPI ranks")
    main(args)
    message("Done.")
