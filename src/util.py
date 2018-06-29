import numpy


def sampling4sphere(p_num):
    """
    Given number of points, distribute evenly on hyper surface of a 4-sphere.
    """
    points = numpy.zeros((2 * p_num, 4))
    dim = 4
    surface_area = dim * numpy.pi ** (dim / 2) / (dim / 2)  # for even N
    delta = numpy.exp(numpy.log(surface_area / p_num) / 3)
    Iter = 0
    ind = 0
    maxIter = 1000
    while ind != p_num and Iter < maxIter:
        ind = 0
        deltaW1 = delta
        w1 = 0.5 * deltaW1
        while w1 < numpy.pi:
            q0 = numpy.cos(w1)
            deltaW2 = deltaW1 / numpy.sin(w1)
            w2 = 0.5 * deltaW2
            while w2 < numpy.pi:
                q1 = numpy.sin(w1) * numpy.cos(w2)
                deltaW3 = deltaW2 / numpy.sin(w2)
                w3 = 0.5 * deltaW3
                while w3 < 2 * numpy.pi:
                    q2 = numpy.sin(w1) * numpy.sin(w2) * numpy.cos(w3)
                    q3 = numpy.sin(w1) * numpy.sin(w2) * numpy.sin(w3)
                    points[ind, :] = numpy.array([q0, q1, q2, q3])
                    ind += 1
                    w3 += deltaW3
                w2 += deltaW2
            w1 += deltaW1
        delta *= numpy.exp(numpy.log(float(ind) / p_num) / 3)
        Iter += 1
    return points[0:p_num, :]
