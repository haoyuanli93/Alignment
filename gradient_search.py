import scipy.ndimage
import scipy.ndimage.interpolation
import numpy
import util, tmp
from mpi4py import MPI
import time
import argparse

########################################################################################################################
# Step 1: Initialize
########################################################################################################################
# Define the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--cat_num', type=int, help="Specify the category to align.")
parser.add_argument('--tag', type=str, help="Specify the tag of the saved file.")

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Parse
args = parser.parse_args()
cat_param = args.cat_num  # Specify which reconstruction to align
input_file = '/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/aligned_{}_coarse.npy'.format(cat_param)
object_shape = numpy.array([27, ] * 3, dtype=numpy.int64)
tag = args.tag

# All node load the two arrays to compare
fixed_target = numpy.load('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/input/chop_complete.npy')
movable_target = numpy.load(input_file)

# Initialize the sampling of rotation axises.
direction_spacing = 0.1
directions, _, dist_min = util.sample_full_2d_sphere(delta_theta=direction_spacing,
                                                     delta_phi=direction_spacing)
# Initialize the sampling of rotation angles
degree_spacing = 0.05
degree_num = 21
degrees = numpy.arange(-10, 10) * degree_spacing

# Initialize the shift list
shift_num = 11
shift_spacing = 1.
shift_list = numpy.arange(-5, 5) * shift_spacing

# All nodes generate a job list and find the portion of jobs belonging to itself.
range_array = util.get_batch_range_list(util.get_batch_num_list(directions.shape[0], batch_num=comm_size - 1))
job_start = range_array[comm_rank, 0]
job_stop = range_array[comm_rank, 1]
job_num = job_stop - job_start

# Create a holder to store all the IoU value
IoU_list = numpy.zeros(job_num * degree_num * shift_num ** 3, dtype=numpy.float64)
print("Node {} needs to calculate {} IoU values".format(comm_rank, IoU_list.shape[0]))

########################################################################################################################
# Step 2: Do one iteration of search
########################################################################################################################
IoU = tmp.calculate_IoU(movable_target=movable_target, fixed_target=fixed_target,
                        job_start=job_start, job_stop=job_stop,
                        directions=directions, degrees=degrees, shift_list=shift_list)
comm.Barrier()
# One node collects all the IoU and find the best result
IoU_data = comm.gather(IoU_list, root=0)

if comm_rank == 0:
    IoU_all = numpy.concatenate(IoU_data)
    # Find the corresponding transformation
    index = numpy.argmax(IoU_all)
    axis, angle, shift = util.recover_the_transform(index, directions, degrees, shift_list, shift_list, shift_list)
    # Calculate the corresponding transformed volume
    transformed = util.rotation_and_shift(obj=movable_target, axis=axis, angle=angle, shift=shift)
    # Save this transformed volume
    numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/aligned_{}_step1.npy'.format(cat_param),
               transformed)

    # After this iteration, replace movable_target with this optimal value.
    movable_target = transformed

# Refresh the movable target across all the nodes.
movable_target = comm.Bcast(obj=movable_target, root=0)
comm.Barrier()

########################################################################################################################
# Step 3: Do several other iterations of search
########################################################################################################################

IoU = tmp.calculate_IoU(movable_target=movable_target, fixed_target=fixed_target,
                        job_start=job_start, job_stop=job_stop,
                        directions=directions, degrees=degrees, shift_list=shift_list)

# Step 2: One node collect all the IoU and find the best result
IoU_data = comm.gather(IoU_list, root=0)

if comm_rank == 0:
    IoU_all = numpy.concatenate(IoU_data)
    # Find the corresponding transformation
    index = numpy.argmax(IoU_all)
    axis, angle, shift = util.recover_the_transform(index, directions, degrees, shift_list, shift_list, shift_list)
    # Calculate the corresponding transformed volume
    transformed = util.rotation_and_shift(obj=movable_target, axis=axis, angle=angle, shift=shift)
    # Save this transformed volume
    numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/aligned_{}_step1.npy'.format(cat_param),
               transformed)
