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
    :param delta_phi:  The expected space between phis
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
