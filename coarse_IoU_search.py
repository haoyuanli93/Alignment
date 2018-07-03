"""
This script implement the brute force search strategy with mpi4py enhancement
"""
import scipy.ndimage
import scipy.ndimage.interpolation
import numpy
import util
from mpi4py import MPI
import time

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Define sampling param
delta_degree = 0.15
translation_range = 10

# Define target parameter
input_file = '/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/input/normalized_sample_1.npy'
tag = 1

# All the node generate a sampling of the directions and degrees to inspect
directions, _, dist_min = util.sample_full_2d_sphere(delta_theta=delta_degree, delta_phi=delta_degree)
degrees = util.sample_full_circle(delta_psi=delta_degree)

# All nodes generate a job list and find the portion of jobs belonging to itself.
range_array = util.get_batch_range_list(util.get_batch_num_list(directions.shape[0], batch_num=comm_size))
job_start = range_array[comm_rank, 0]
job_stop = range_array[comm_rank, 1]
job_num = job_stop - job_start

# Create a holder to store all the IoU value
degree_num = degrees.shape[0]
IoU_list = numpy.zeros(job_num * degree_num * (2 * translation_range) ** 3, dtype=numpy.float64)
print("Node {} needs to calculate {} IoU values".format(comm_rank, IoU_list.shape[0]))

# All node load the two arrays to compare
fixed_target = numpy.load('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/input/complete.npy')
movable_target = numpy.load(input_file)

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
        offset = numpy.array([63, ] * 3) - rotation_matrix.dot(numpy.array([63, ] * 3))

        # Rotate the sample space
        scipy.ndimage.affine_transform(input=movable_target,
                                       matrix=rotation_matrix,
                                       offset=offset,
                                       output_shape=(128, 128, 128),
                                       output=rotated,
                                       order=1,
                                       mode='constant', cval=0.0, prefilter=True)

        # Loop through all spacial shift
        for shift_x in range(-translation_range, translation_range):
            for shift_y in range(-translation_range, translation_range):
                for shift_z in range(-translation_range, translation_range):
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
    toc = time.time()
    if comm_rank == 0:
        print("It takes {} seconds to calculate all IoUs for a single axis.".format(toc - tic))
comm.Barriar()
# Step 2: One node collect all the IoU and find the best result
IoU_data = comm.gather(IoU_list, root=0)

if comm_rank == 0:
    IoU_all = numpy.concatenate(IoU_data)
    numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/IoU_all_{}.npy'.format(tag), IoU_all)
