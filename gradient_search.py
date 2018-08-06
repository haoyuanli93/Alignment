import numpy, os, h5py
import util, tmp
from mpi4py import MPI
import time, datetime
import argparse

########################################################################################################################
# Step 1: Initialize
########################################################################################################################
# Define the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, help="Specify the output folder.")
parser.add_argument('--movable_target', type=str, help="Specify the movable target.")
parser.add_argument('--fixed_target', type=str, help="Specify the fixed target.")
parser.add_argument('--tag', type=str, help="Some tag to distinguish this result from the others.")

# Initialize the MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# Parse
args = parser.parse_args()
input_fixed = args.fixed_target
input_movable = args.movable_target
output_address = args.output
tag = args.tag

# Check the parameters
if comm_rank == 0:
    # The first node check if the output folder exist
    if not os.path.isdir(output_address):
        # Create a folder in the output address
        os.makedirs(output_address)

    # Create a time stamp
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    # Prepare the file name
    if tag:
        file_name = output_address + '/gradient_output_{}_{}.h5'.format(tag, stamp)
    else:
        file_name = output_address + '/gradient_output_{}.h5'.format(stamp)

    # Create a h5py file to hold the results
    with h5py.File(file_name, 'w') as h5file:
        h5file.create_dataset("fixed_target", data=input_fixed)
        h5file.create_dataset("movable_target", data=input_movable)

comm.Barrier()

# Load the files and check the shape
fixed_target = numpy.load(input_fixed)
movable_target = numpy.load(input_movable)
object_shape = movable_target.shape

# Generate a searching precision list
"""
Due to my current setup, the searching precision is decided by the spatial precision and the angular precision
of the rotation angle.
"""
iteration_number = 5

# specify some parameter concerning the searching of the rotation and translation.
degree_num = 8
shift_num = 5

degree_range_list = numpy.square(2. / numpy.arange(1, iteration_number + 1))
shift_range_list = 5 / numpy.arange(1, iteration_number + 1)

# Initialize the sampling of rotation axises.
direction_spacing = 0.2
directions, _, dist_min = util.sample_full_2d_sphere(delta_theta=direction_spacing, delta_phi=direction_spacing)

# All nodes generate a job list and find the portion of jobs belonging to itself.
range_array = util.get_batch_range_list(util.get_batch_num_list(directions.shape[0], batch_num=comm_size))
job_start = range_array[comm_rank, 0]
job_stop = range_array[comm_rank, 1]
job_num = job_stop - job_start

# Create a holder to store all the IoU value
IoU_list = numpy.zeros(job_num * (2 * degree_num + 1) * (2 * shift_num + 1) ** 3, dtype=numpy.float64)
print("Node {} needs to calculate {} IoU values".format(comm_rank, IoU_list.shape[0]))

# Create a new variable to check the calculation time
times_start = []
times_stop = []

########################################################################################################################
# Step 2: Searching all the directions and rotation angles and shifts
########################################################################################################################
for l in range(iteration_number):
    # Generate the degree list and shift list to investigate
    degrees, shift_list = tmp.get_degree_and_shift_list(degree_num=degree_num, degree_range=degree_range_list[l],
                                                        shift_num=shift_num, shift_range=shift_range_list[l])
    tic = time.time()
    times_start.append(tic)
    # Each node calculate the corresponding IoU values
    IoU_list = tmp.calculate_iou(movable_target=movable_target, fixed_target=fixed_target,
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
        transformed, _axis, _angle, _shift, _iou = tmp.to_optimal_orientation(iou_all=IoU_all,
                                                                              movable_target=movable_target,
                                                                              directions=directions,
                                                                              degrees=degrees,
                                                                              shift_list=shift_list)

        # Save this transformed volume
        with h5py.File(file_name, 'r+') as h5file:
            grp = h5file.create_group("step_{}".format(l))
            # grp.create_dataset('iou_all', data=IoU_all)
            grp.create_dataset('volume', data=transformed)
            grp.create_dataset('axis', data=_axis)
            grp.create_dataset('angle', data=_angle)
            grp.create_dataset('shift', data=_shift)
            grp.create_dataset('iou', data=_iou)

        # After this iteration, replace movable_target with this optimal value.
        movable_target = transformed

    comm.Barrier()
    # Refresh the movable target across all the nodes.
    comm.Bcast(movable_target, root=0)
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
