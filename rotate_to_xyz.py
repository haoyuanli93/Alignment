import numpy
import argparse

import util

# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help="Specify the input file.")
parser.add_argument("--tag", type=str, help="Use the tag to distinguish different rotated patterns.")

# Parse
args = parser.parse_args()
input_file = args.input_file
tag = args.tag

# Define parameters
output_folder = '../output/'
distance_range = [5, 40]
grid_size = tuple([125, ] * 3)
mesh_shape = tuple([125, 125, 125, 3])
x_step = numpy.arange(start=-62, stop=63, dtype=numpy.float64)
y_step = numpy.arange(start=-62, stop=63, dtype=numpy.float64)
z_step = numpy.arange(start=-62, stop=63, dtype=numpy.float64)

# Create the coordinate mesh
x_map, y_map, z_map = numpy.meshgrid(x_step, y_step, z_step)

# Calculate the length mesh
length_mesh = util.get_distance_mesh(x_map=x_map, y_map=y_map, z_map=z_map)

# Calculate the range grid
range_holder = (length_mesh <= distance_range[1]) & (length_mesh >= distance_range[0])
pixel_num = numpy.sum(range_holder)

# Load data
data = numpy.fromfile(input_file, dtype=numpy.float64).reshape(grid_size)

# Extract the data and coordinate mesh within the range
data_in_range = data[range_holder]
x_in_range = x_map[range_holder]
y_in_range = y_map[range_holder]
z_in_range = z_map[range_holder]

# Calculate the rotational inertia
tensor = util.get_rotational_inertia(data=data_in_range,
                                     x=x_in_range,
                                     y=y_in_range,
                                     z=z_in_range,
                                     pixel_num=pixel_num)

# Calculate the eigensystem
eigval, eigvec = numpy.linalg.eigh(tensor)

# Save the eigenvectors and eigenvalues
numpy.save(output_folder + 'eigval_{}.npy'.format(tag), eigval)
numpy.save(output_folder + 'eigvec_{}.npy'.format(tag), eigvec)

# Rotate the space
coordinate_holder = numpy.zeros((numpy.prod(grid_size), 3), dtype=numpy.float64)
coordinate_holder[:, 0] = x_map.reshape(numpy.prod(grid_size))
coordinate_holder[:, 1] = y_map.reshape(numpy.prod(grid_size))
coordinate_holder[:, 2] = z_map.reshape(numpy.prod(grid_size))

value_holder = data.reshape(numpy.prod(grid_size))

rotated_data = util.rotate_the_space(coordinate_map=coordinate_holder,
                                     matrix=eigvec,
                                     value_map=value_holder,
                                     output_shape=grid_size)

# Save the rotated result
numpy.save(output_folder + 'rotated_{}.npy'.format(tag), rotated_data)
