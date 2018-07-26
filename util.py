import numpy as np
import scipy.ndimage as sn


def get_radial_distribution(pattern, category_map, number_of_interval):
    """
    Return the radial distribution based on the category_map. 
    :param: pattern: This is the dataset to inspect. Notice that this has to be the pattern stack obtained from det.calib rather than det.image.
    :param: category_map : The category map of the pixel.
    :param: number_of_interval: The number of intervals.
    :return: A numpy array containing the radial distriabution. The shape is (number_of_interval,)
    """
    distribution = np.zeros(number_of_interval)
    # Calculate the distribution
    for category_idx in range(number_of_interval):
        distribution[category_idx] = np.mean(pattern[category_map == category_idx])

    return distribution


def get_pixel_map(values, ends, output_mode="in situ"):
    """

    Input:

    values : numpy array, values that are used to classify the indexes. 

    ends :  (M,2)-shaped numpy array. Contain the end points of each category.
            There will be M categories. At present, the interval is left open,
            and right close.

    "output_mode": String. When output_mode=="per class", the output will be of
            such shape (M, shape of "values"). When output_mode=="in situ", the
            output will be of the shape of the variable "values". Each site in 
            the output numpy array will carry a value in [0,1,2,...,M-1,M]. This 
            indicates of the specific site. Notice that there are M+1 values rather
            than M values. This is because that it is possible to have sites that
            are not in any classes. They are assigned the value M.

    Output:

    A numpy array of the shape of the variable "values" or of the shape 
    (M, the shape of "values") depending on the value of the variable "output_mode".

    """

    # Get the structure information of the input variable
    _values_shape = values.shape
    _category_number = ends.shape[0]

    if output_mode == "per class":
        # Create the mapping variable
        _class_per_site = np.zeros((_category_number + 1,) + _values_shape, dtype=bool)
        # Create a holer for simplicity
        _holder = np.zeros_like(values, dtype=bool)

        for l in range(_category_number):
            # Assign values to the mapping
            _holder[(values > ends[l, 0]) & (values <= ends[l, 1])] = True
            _class_per_site[l, :] = np.copy(_holder)

        # Get the value for the last class
        """
        Because the summation of all the boolean along the first dimension should be one. 
        The value of the last class is one minus the value of the summation of all the value
        along the first dimension

        Because the variable is initialized as zero. We can also do the summation including
        The last category.
        """

        _class_per_site[_category_number] = np.logical_not(np.sum(_class_per_site, axis=0))

        return _class_per_site

    elif output_mode == "in situ":
        # Create the mapping variable.
        _class_in_situ = np.ones_like(values, dtype=np.int32) * _category_number

        for l in range(_category_number):
            _class_in_situ[(values > ends[l, 0]) & (values <= ends[l, 1])] = l

        return _class_in_situ

    else:
        raise Exception("The value of the output_mode is invalid. Please use either \'in situ\' or \'per class\'. ")


def get_distance_mesh(x_map, y_map, z_map):
    """
    Get the length of the pixel with respect to the origin defined by the x_map, y_map and z_map
    :param x_map: The x coordinate of each pixel.
    :param y_map: The y coordinate of each pixel
    :param z_map: The z coordinate of each pixel
    :return: The distance of each pixel from the origin defined by the coordinate maps.
    """
    return np.sqrt(np.square(x_map) + np.square(y_map) + np.square(z_map))


def get_rotational_inertia(data, x, y, z, pixel_num):
    inertia = np.zeros((3, 3), dtype=np.float64)

    # Calculate the Ixx term
    inertia[0, 0] = np.sum(np.multiply(data,
                                       np.multiply(y, y) + np.multiply(z, z)))

    # Calculate the Ixy and Iyx term
    inertia[0, 1] = - np.sum(np.multiply(data, np.multiply(x, y)))
    inertia[1, 0] = inertia[0, 1]

    # Calculate the Ixz and Izx term
    inertia[0, 2] = - np.sum(np.multiply(data, np.multiply(x, y)))
    inertia[2, 0] = inertia[0, 2]

    # Calculate the Iyy term
    inertia[1, 1] = np.sum(np.multiply(data,
                                       np.multiply(x, x) + np.multiply(z, z)))

    # Calculate the Iyz and Izy term
    inertia[1, 2] = - np.sum(np.multiply(data, np.multiply(y, z)))
    inertia[2, 1] = inertia[1, 2]

    # Calculate the Izz term
    inertia[2, 2] = np.sum(np.multiply(data,
                                       np.multiply(x, x) + np.multiply(y, y)))

    return inertia / float(pixel_num)


def rotate_the_space(coordinate_map, matrix, value_map, output_shape):
    """
    Calculate the rotated image.

    :param coordinate_map: The coordinate of the original image. [[x1,y1,z1],
                                                                  [x2,y2,z2],
                                                                  ...]
    :param matrix: The rotation matrix
    :param value_map: The value of for each pixel.
    :param output_shape: The shape of the output array.
    :return: The value at each pixel in the rotated space.
    """
    rotated_position = matrix.dot(coordinate_map)
    interpolated_values = sn.map_coordinates(input=value_map,
                                             coordinates=rotated_position,
                                             order=1,
                                             mode='constant',
                                             cval=0.0)
    return interpolated_values.reshape(output_shape)


# Sample through the rotation group SO(3)
def angle_axis_to_mat(axis, theta):
    """
    Convert rotation with angle theta around a certain axis to a rotation matrix in 3D.
    """
    if len(axis) is not 3:
        raise ValueError('Number of axis element must be 3!')
    axis = axis.astype(float)
    axis /= np.linalg.norm(axis)
    a = axis[0]
    b = axis[1]
    c = axis[2]
    cos_theta = np.cos(theta)
    bracket = 1 - cos_theta
    a_bracket = a * bracket
    b_bracket = b * bracket
    c_bracket = c * bracket
    sin_theta = np.sin(theta)
    asin_theta = a * sin_theta
    bsin_theta = b * sin_theta
    csin_theta = c * sin_theta
    rot3d = np.array([[a * a_bracket + cos_theta, a * b_bracket - csin_theta, a * c_bracket + bsin_theta],
                      [b * a_bracket + csin_theta, b * b_bracket + cos_theta, b * c_bracket - asin_theta],
                      [c * a_bracket - bsin_theta, c * b_bracket + asin_theta, c * c_bracket + cos_theta]])
    return rot3d


def sample_full_2d_sphere(delta_theta, delta_phi):
    """
    Try to sample the 2d sphere as uniform as possible.
    Notice that this is not a truly uniform sampling. But it's easy to implement to understand and can yield pretty
    good sampling.

    :param delta_theta: The expected space between thetas
    :param delta_phi:  The expected space between phis on the sphere. Notice that this is a distance rather than
                        the actual angle. Or equivalently, you may think of this as the angle difference along
                        the equator.
    :return: The sampled result, the largest distances, the smallest distances.
    """
    # Create a holder for the result
    holder = []

    # Calculate how many circles to sample
    circle_num = int(np.pi / delta_theta + 1)
    # The actual space between the sampled points of theta
    theta_step = np.pi / circle_num

    # There are singularities at the poles, so they are treated separately.
    for l in range(circle_num + 1):
        # The length of the circle for the sampling of phi
        length = 2 * np.pi * np.sin(l * theta_step)

        # Deal with the poles
        if length < delta_phi:
            tmp = np.zeros((1, 3), dtype=np.float64)
            tmp[0, 2] = np.sign(np.cos(l * theta_step))
            # Save this result to the holder
            holder.append(tmp)
            continue

        # The actual sample number for phi along this circle
        phi_num = int(length / delta_phi)
        # The actual space between the sampled points of phi
        phi_step = 2 * np.pi / phi_num
        # Calculate all the sampled phis
        phi_list = np.arange(start=0, stop=phi_num, dtype=np.float64) * phi_step

        # Create a tmp holder to store the points on this circle
        tmp = np.ones((phi_num, 3), dtype=np.float64)
        tmp[:, 0] = np.sin(l * theta_step) * np.cos(phi_list)
        tmp[:, 1] = np.sin(l * theta_step) * np.sin(phi_list)
        tmp[:, 2] = np.cos(l * theta_step)

        # Save this result to the holder
        holder.append(tmp)

    # Chain together all the sampled directions
    sample_list = np.concatenate(holder, axis=0)

    # Distance
    distance = np.sqrt(2.00001 - 2 * np.dot(sample_list, sample_list.T))
    largest_distance = np.max(distance)

    # Remove the diagonal values
    np.fill_diagonal(distance, 5)
    smallest_distance = np.min(distance)

    return sample_list, largest_distance, smallest_distance


def sample_full_circle(delta_psi):
    """
    Sample the circle S1 uniformly

    :param delta_psi: The expected space between psi
    :return: The sampled result.
    """

    num = int(np.pi * 2 / delta_psi + 1)
    return np.linspace(start=0, stop=np.pi * 2, endpoint=False, num=num)


def sample_northern_pole(delta_theta, delta_phi, theta_range):
    """
    Try to sample a circular strip around the northern pole.

    Notice that this is not a truly uniform sampling. But it's easy to implement to understand and can yield pretty
    good sampling.

    :param delta_theta: The expected space between thetas
    :param delta_phi:  The expected space between phis on the sphere. Notice that this is a distance rather than
                        the actual angle. Or equivalently, you may think of this as the angle difference along
                        the equator.
    :param theta_range: [The start of the searching region, the end of the searching region] a list or an array.
    :return: The sampled result, the largest distances, the smallest distances.
    """

    # Check the parameter validity
    if (theta_range[0] < 0) or (theta_range[1] <= theta_range[0]):
        raise Exception("Please check the following:"
                        "1. The value of both entry of theta range has to be within [0, pi]."
                        "2. The second value in theta_range has to be larger than the first value.")

    # Create a holder for the result
    holder = []

    # Calculate how many circles to sample
    circle_num = int(np.abs(theta_range[1] - theta_range[0]) / delta_theta + 1)
    # The actual space between the sampled points of theta
    theta_list = np.linspace(start=theta_range[0], stop=theta_range[1], num=circle_num, endpoint=True)

    # There are singularities at the poles, so they are treated separately.
    for theta in theta_list:
        # The length of the circle for the sampling of phi
        length = 2 * np.pi * np.sin(theta)

        # Deal with the poles
        if length < delta_phi:
            tmp = np.zeros((1, 3), dtype=np.float64)
            tmp[0, 2] = np.sign(np.cos(theta))
            # Save this result to the holder
            holder.append(tmp)
            continue

        # The actual sample number for phi along this circle
        phi_num = int(length / delta_phi)
        # The actual space between the sampled points of phi
        phi_step = 2 * np.pi / phi_num
        # Calculate all the sampled phis
        phi_list = np.arange(start=0, stop=phi_num, dtype=np.float64) * phi_step

        # Create a tmp holder to store the points on this circle
        tmp = np.ones((phi_num, 3), dtype=np.float64)
        tmp[:, 0] = np.sin(theta) * np.cos(phi_list)
        tmp[:, 1] = np.sin(theta) * np.sin(phi_list)
        tmp[:, 2] = np.cos(theta)

        # Save this result to the holder
        holder.append(tmp)

    # Chain together all the sampled directions
    sample_list = np.concatenate(holder, axis=0)

    # Distance
    distance = np.sqrt(2.00001 - 2 * np.dot(sample_list, sample_list.T))
    largest_distance = np.max(distance)

    # Remove the diagonal values
    np.fill_diagonal(distance, 5)
    smallest_distance = np.min(distance)

    return sample_list, largest_distance, smallest_distance


def sample_part_of_circle(delta_psi, psi_range):
    """
    Sample a small portion of S1 uniformly.

    :param delta_psi: The expected space between psi
    :param psi_range: [The start of the searching region, the end of the searching region] a list or an array.
    :return: The sampled result.
    """

    num = int(np.abs(psi_range[1] - psi_range[0]) / delta_psi + 1)
    return np.linspace(start=psi_range[0], stop=psi_range[1], endpoint=True, num=num)


def get_batch_num_list(total_num, batch_num):
    """
    Generate a list containing the data number per batch.
    The idea is that the difference between each batches is at most one pattern.
    :param total_num: The total number of patterns.
    :param batch_num: The number of batches to build.
    :return: A list containing the data number in each batch.
    """

    redundant_num = np.mod(total_num, batch_num)
    if redundant_num != 0:
        number_per_batch = total_num // batch_num
        batch_num_list = [number_per_batch + 1, ] * redundant_num
        batch_num_list += [number_per_batch, ] * (batch_num - redundant_num)
    else:
        number_per_batch = total_num // batch_num
        batch_num_list = [number_per_batch, ] * batch_num

    return batch_num_list


def get_batch_range_list(batch_num_list):
    """
    Short hand to caonvet the batch_num_list to batch_range_list

    :param batch_num_list: The number of batches to build.
    :return: A numpy array containing the batch range for each range.
                [
                [start, end],
                [start, end],
                ...
                ]
    """

    holder = np.zeros((len(batch_num_list), 2), dtype=np.int64)
    tmp = np.cumsum([0, ] + batch_num_list)

    holder[:, 0] = tmp[:-1]
    holder[:, 1] = tmp[1:]

    return holder


def recover_the_transform(idx_to_inspect, axis_list, degree_list, shiftx_list, shifty_list, shiftz_list):
    """
    Recover the transformation based on the info of the index of the IoU list.

    :param idx_to_inspect: The index to inspect
    :param axis_list:  The numpy array containing all the rotation axis
    :param degree_list: The numpy array containing all the rotation degrees
    :param shiftx_list: The numpy array containing all shift along x direction.
    :param shifty_list:  The numpy array containing all shift along y direction.
    :param shiftz_list:  The numpy array containing all shift along z direction.
    :return: rotation axis, rotation degree, [shift vector x,y,z]
    """

    # Get the length of each list
    degree_num = degree_list.shape[0]
    x_num = shiftx_list.shape[0]
    y_num = shifty_list.shape[0]
    z_num = shiftz_list.shape[0]

    # Calculate the shift along z
    z_idx = np.mod(idx_to_inspect, z_num).astype(np.int64)
    # Modify the index
    idx_to_inspect = np.floor_divide(idx_to_inspect, z_num, dtype=np.int64)

    # Calculate the shift along y
    y_idx = np.mod(idx_to_inspect, y_num).astype(np.int64)
    # Modify the index
    idx_to_inspect = np.floor_divide(idx_to_inspect, y_num, dtype=np.int64)

    # Calculate the shift along x
    x_idx = np.mod(idx_to_inspect, x_num).astype(np.int64)
    # Modify the index
    idx_to_inspect = np.floor_divide(idx_to_inspect, x_num, dtype=np.int64)

    # Calculate the degree idx
    degree_idx = np.mod(idx_to_inspect, degree_num).astype(np.int64)
    # Modify the index
    idx_to_inspect = np.floor_divide(idx_to_inspect, degree_num, dtype=np.int64)

    # Calculate the axis idx
    axis_idx = int(idx_to_inspect)

    return axis_list[axis_idx], degree_list[degree_idx], [shiftx_list[x_idx], shifty_list[y_idx], shiftz_list[z_idx]]


def get_mass_center(obj):
    """
    Calculate the mass center of the object in the space. The origin is at [0,0,0] position
    :param obj: A 3D numpy array containing the density
    :return: A 1D numpy array center_of_mass = [a,b,c]
    """
    # Create a coordinate grid
    space_size = obj.shape
    coor_grid = np.meshgrid(np.arange(space_size[0]), np.arange(space_size[1]), np.arange(space_size[2]))

    # Calculate the mass center
    holder = np.zeros(3)
    for axis in range(3):
        holder[axis] = np.sum(np.multiply(coor_grid[axis], obj)) / np.sum(obj)

    return holder


def rotation_and_shift(obj, axis, angle, shift):
    """
    This function calculate the transformed volume. One first performs the rotation defined by the axis and the angle
    with respect to the center of mass of the space and then shift the space according to shift.

    :param obj: A 3D numpy array containing the density
    :param axis: The axis of rotation.
    :param angle: The angle of rotation around that axis
    :param shift: The displacement vector of the shift operation.
    :return: The transformed volume. A 3D numpy array.
    """
    # Obtain the center of mass of the movable target
    center = get_mass_center(obj)

    # Calculate the affine map to use scipy.ndimage.affine_transform
    rotation_matrix = angle_axis_to_mat(axis=axis, theta=angle)
    offset = center - rotation_matrix.dot(center)

    # Rotate the sample space
    rotated = sn.affine_transform(input=obj,
                                  matrix=rotation_matrix,
                                  offset=offset,
                                  order=1,
                                  mode='constant', cval=0.0, prefilter=True)

    # shift
    shifted = sn.interpolation.shift(input=rotated,
                                     shift=shift,
                                     order=1,
                                     mode='constant',
                                     cval=0.0,
                                     prefilter=True)

    return shifted


def get_IoU_inplace(shifted, fixed, cap_holder, cup_holder):
    """
    Get the intersection over union value of the two objects. The situation is my mind for this function is
    that we have a fixed object, and a shifted object. The shift is small. We want to calculate the IoU
    a lot of times. Therefore, two holders are created to hold the intersection and the union space.

    :param shifted: The shifted object.
    :param fixed: The fixed object.
    :param cap_holder: The numpy array to hold the intersection of the two object.
    :param cup_holder: The numpy array to hold the union of the two object
    :return: The IoU value.
    """

    # Calculate the intersection
    np.minimum(shifted, fixed, out=cap_holder)
    # Calculate the union
    np.maximum(shifted, fixed, out=cup_holder)

    return np.sum(cap_holder) / np.sum(cup_holder)
