import numpy as np

from ... import maps
from ..utils import omega
from .utils.source_utils import homo1DModelSource
import discretize
from discretize.utils import volume_average
from ..natural_source.sources import PlanewaveXYPrimary

#################
#    Sources    #
#################

class PlanewaveXYPrimary(PlanewaveXYPrimary):
    """
    WEM planewave source for both polarizations (x and y)
    estimated from a single 1D primary models.

    :param list receiver_list: List of SimPEG.electromagnetics.wireless_electro_magnetic.receivers.PointWEM
    :param float frequency: frequency for the source
    """

    # This class is herited from SimPEG.elextromagnetics.natural_source.sources.
    def ePrimary(self, simulation):
        if self._ePrimary is None:
            sigma_1d, _ = self._get_sigmas(simulation)
            self._ePrimary = homo1DModelSource(
                simulation.mesh, self.frequency, sigma_1d
            )
        return self._ePrimary
