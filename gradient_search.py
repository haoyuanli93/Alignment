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

# Generate a searching precision list
"""
Due to my current setup, the searching precision is decided by the spatial precision and the angular precision
of the rotation angle.
"""
iteration_number = 9
degree_spacing_list = numpy.power(0.5, numpy.arange(1, 10)) / 10.
shift_spacing_list = numpy.power(0.5, numpy.arange(0, 9))

# All node load the two arrays to compare
fixed_target = numpy.load('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/input/chop_complete.npy')
movable_target = numpy.load(input_file)

# Initialize the sampling of rotation axises.
direction_spacing = 0.1
directions, _, dist_min = util.sample_full_2d_sphere(delta_theta=direction_spacing, delta_phi=direction_spacing)

# specify some parameter concerning the searching of the rotation and translation.
degree_num = 21
shift_num = 11

# All nodes generate a job list and find the portion of jobs belonging to itself.
range_array = util.get_batch_range_list(util.get_batch_num_list(directions.shape[0], batch_num=comm_size - 1))
job_start = range_array[comm_rank, 0]
job_stop = range_array[comm_rank, 1]
job_num = job_stop - job_start

# Create a holder to store all the IoU value
IoU_list = numpy.zeros(job_num * degree_num * shift_num ** 3, dtype=numpy.float64)
print("Node {} needs to calculate {} IoU values".format(comm_rank, IoU_list.shape[0]))

# Create a new variable to check the calculation time
times_start = []
times_stop = []

########################################################################################################################
# Step 2: Searching all the directions and rotation angles and shifts
########################################################################################################################
for l in range(iteration_number):
    # Generate the degree list and shift list to investigate
    degrees, shift_list = tmp.get_degree_and_shift_list(degree_num=degree_num, degree_spacing=degree_spacing_list[l],
                                                        shift_num=shift_num, shift_spacing=shift_spacing_list[l])
    tic = time.time()
    times_start.append(tic)
    # Each node calculate the corresponding IoU values
    IoU = tmp.calculate_iou(movable_target=movable_target, fixed_target=fixed_target,
                            job_start=job_start, job_stop=job_stop,
                            directions=directions, degrees=degrees, shift_list=shift_list)
    toc = time.time()
    times_stop.append(toc)
    print("Node {} takes {:.2f} seconds to calculate {} IoU values.".format(comm_rank, toc - tic, IoU_list.shape[0]))
    comm.Barrier()

    # One node collects all the IoU and find the best result
    IoU_data = comm.gather(IoU_list, root=0)

    if comm_rank == 0:
        IoU_all = numpy.concatenate(IoU_data)
        transformed = tmp.transform_to_the_optimal_orientation(iou_all=IoU_all, movable_target=movable_target,
                                                               directions=directions, degrees=degrees,
                                                               shift_list=shift_list)
        # Save this transformed volume
        numpy.save('/reg/d/psdm/amo/amo86615/res/haoyuan/alignment/output/aligned_{}_step{}.npy'.format(cat_param, l),
                   transformed)

        # After this iteration, replace movable_target with this optimal value.
        movable_target = transformed

    # Refresh the movable target across all the nodes.
    movable_target = comm.Bcast(obj=movable_target, root=0)
    comm.Barrier()

########################################################################################################################
# Step 3: Do several other iterations of search
########################################################################################################################
# Clean up the calculation. Each node report the average calculation time of IoU
total_time = numpy.sum(numpy.array(times_stop) - numpy.array(times_start))
total_num = iteration_number * IoU_list.shape[0]

content = "In total, node {} spends {:.2f} seconds to calculate {} IoU values".format(comm_rank, total_time, total_num)
content += "\n"
content += "The average calculation time per 1000 IoU values is {:.2f}".format(total_time / total_num * 1000)
print(content)
