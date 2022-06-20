from ..natural_source.receivers import PointNaturalSource

#################
#   Receivers   #
#################

class PointWEM(PointNaturalSource):
    """
    WEM source receiver base class.

    Assumes that the data locations are xyz coordinates.

    :param numpy.ndarray locs: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    :param string component: real, imaginary, phase or apparent resistivity component 'real', 'imag', 'phase' or 'apparent_resistivity' 
    """
    pass