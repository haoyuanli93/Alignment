"""
This script implement the brute force search strategy with mpi4py enhancement
"""

import numpy
import util
import h5py
from mpi4py import MPI

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Define sampling param
delta_degree = 0.15
translation_range = 10

# All the node generate a sampling of the directions and degrees to inspect
directions, _, dist_min = util.sample_full_2d_sphere(delta_theta=delta_degree, delta_phi=delta_degree)
degrees = util.sample_full_circle(delta_psi=delta_degree)

# All nodes generate a job list and find the portion of jobs belonging to itself.
range_array = util.get_batch_range_list(util.get_batch_num_list(directions.shape[0], batch_num=comm_size))
job_start = range_array[comm_rank, 0]
job_stop = range_array[comm_rank, 1]

# All node load the two arrays to compare

