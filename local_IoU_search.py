"""
This script implement the brute force search strategy with mpi4py enhancement

This script performs the coarse searching. The alignment is carried out with
respect to the reconstruction with all diffraction patterns.
"""
import scipy.ndimage
import scipy.ndimage.interpolation
import numpy
import util
from mpi4py import MPI
import time
import argparse

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--cat_num', type=int, help="Specify the category to align.")
parser.add_argument('--tag', type=str, help="Specify the tag of the saved file.")

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Define sampling param
delta_degree = 0.075
angle_range = 1.
shift_list = numpy.linspace(start=-2, stop=2, num=10, endpoint=True)
shift_num = shift_list.shape[0]

# Define target parameter
# Parse
args = parser.parse_args()
cat_param = args.cat_num  # Specify which reconstruction to align
input_file = '/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/alignment/output/aligned_{}_coarse.npy'.format(cat_param)
object_shape = numpy.array([27, ] * 3, dtype=numpy.int64)
tag = args.tag

# All the node generate a sampling of the directions and degrees to inspect
directions, _, dist_min = util.sample_northern_pole(delta_theta=delta_degree,
                                                    delta_phi=delta_degree,
                                                    theta_range=angle_range)
degrees = util.sample_part_of_circle(delta_psi=delta_degree, psi_range=angle_range)

# All nodes generate a job list and find the portion of jobs belonging to itself.
range_array = util.get_batch_range_list(util.get_batch_num_list(directions.shape[0], batch_num=comm_size))
job_start = range_array[comm_rank, 0]
job_stop = range_array[comm_rank, 1]
job_num = job_stop - job_start

# Create a holder to store all the IoU value
degree_num = degrees.shape[0]
IoU_list = numpy.zeros(job_num * degree_num * shift_num ** 3, dtype=numpy.float64)
print("Node {} needs to calculate {} IoU values".format(comm_rank, IoU_list.shape[0]))

# All node load the two arrays to compare
fixed_target = numpy.load('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/input/chop_complete.npy')
movable_target = numpy.load(input_file)

# Obtain the center of mass of the movable target
center = util.get_mass_center(movable_target)

# Create holders which will be reused again and again
rotated = numpy.zeros_like(movable_target)
shifted = numpy.zeros_like(movable_target)
intersection = numpy.zeros_like(movable_target)
union = numpy.zeros_like(movable_target)

# Create a variable to account for the index of the result
tmp_idx = 0
# Loop through rotations and translations
for axis_idx in range(job_start, job_stop):
    # Calculate the time
    tic = time.time()
    for degree in degrees:
        # Calculate the affine map to use scipy.ndimage.affine_transform
        rotation_matrix = util.angle_axis_to_mat(axis=directions[axis_idx], theta=degree)
        offset = center - rotation_matrix.dot(center)

        # Rotate the sample space
        scipy.ndimage.affine_transform(input=movable_target,
                                       matrix=rotation_matrix,
                                       offset=offset,
                                       output=rotated,
                                       order=1,
                                       mode='constant', cval=0.0, prefilter=True)

        # Loop through all spacial shift
        for shift_x in shift_list:
            for shift_y in shift_list:
                for shift_z in shift_list:
                    # Shift the rotated space
                    scipy.ndimage.interpolation.shift(input=rotated,
                                                      shift=numpy.array([shift_x, shift_y, shift_z],
                                                                        dtype=numpy.float64),
                                                      output=shifted,
                                                      order=1,
                                                      mode='constant',
                                                      cval=0.0,
                                                      prefilter=True)

                    # Calculate the intersection
                    numpy.minimum(shifted, fixed_target, out=intersection)
                    # Calculate the union
                    numpy.maximum(shifted, fixed_target, out=union)

                    # Calculate the IoU
                    IoU_list[tmp_idx] = numpy.sum(intersection) / numpy.sum(union)

                    # Update the index
                    tmp_idx += 1

    toc = time.time()
    if comm_rank == 0:
        print("It takes {} seconds to calculate all IoUs for a single axis.".format(toc - tic))

comm.Barrier()
# Step 2: One node collect all the IoU and find the best result
IoU_data = comm.gather(IoU_list, root=0)

if comm_rank == 0:
    IoU_all = numpy.concatenate(IoU_data)
    # Save all the IoU values for independent varifications
    numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/IoU_all_{}_{}.npy'.format(cat_param, tag),
               IoU_all)
    # Save all the translations for independent varifications
    numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/axis_all_{}_{}.npy'.format(cat_param, tag),
               directions)
    numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/angle_all_{}_{}.npy'.format(cat_param, tag),
               degrees)
    numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/shift_all_{}_{}.npy'.format(cat_param, tag),
               shift_list)

    # Find the corresponding transformation
    index = numpy.argmax(IoU_all)
    axis, angle, shift = util.recover_the_transform(index, directions, degrees, shift_list, shift_list, shift_list)
    # Calculate the corresponding transformed volume
    transformed = util.rotation_and_shift(obj=movable_target, axis=axis, angle=angle, shift=shift)
    # Save this transformed volume
    numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/aligned_{}_{}.npy'.format(cat_param, tag),
               transformed)
