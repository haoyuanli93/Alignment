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
