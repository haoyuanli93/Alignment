"""
I am currently developing the iterative search method. This requires me to put a lot of
operations into functions whether it's efficient or not.
"""
import scipy.ndimage
import scipy.ndimage.interpolation
import numpy
import util


def calculate_IoU(movable_target, fixed_target, job_start, job_stop, directions, degrees, shift_list):
    """
    This function is a simple abbreviation of the existing local searching procedure.

    :param movable_target: The object that will be aligned with the fixed target.
    :param fixed_target: The object to be aligned with.
    :param job_start: The starting index of the directions variable
    :param job_stop: The stop index of the direction variable
    :param directions: The axis of rotations to investigate.
    :param degrees: The degrees of rotations to investigate.
    :param shift_list: The spatial shift to investigate.
    :return: The IoU value between the movable target and the fixed target after each transformation.
    """
    # Create a holder for all the IoU values.
    IoU_list = numpy.zeros((job_stop - job_start) * (degrees.shape[0]) * (shift_list.shape[0]) ** 3,
                           dtype=numpy.float64)

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
    return IoU_list


def find_the_optimal_position():
    pass
