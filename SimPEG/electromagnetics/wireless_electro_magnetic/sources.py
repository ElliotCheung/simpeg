from .utils.source_utils import homo1DModelSource
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
    def __init__(self, receiver_list, frequency, sigma_primary=None, trans_r=5e5, h_air=1e5, sig_iono=1e-4, qwe_order=40):
        # assert mkvc(self.mesh.h[2].shape,1) == mkvc(sigma1d.shape,1),'The number of values in the 1D background model does not match the number of vertical cells (hz).'
        self.sigma1d = None
        self._sigma_primary = sigma_primary
        self.trans_r = trans_r
        self.h_air = h_air
        self.sig_iono = sig_iono
        self.qwe_order = qwe_order
        super(PlanewaveXYPrimary, self).__init__(receiver_list, frequency)

    def ePrimary(self, simulation):
        if self._ePrimary is None:
            sigma_1d, _ = self._get_sigmas(simulation)
            self._ePrimary = homo1DModelSource(
                simulation.mesh, sigma_1d, self.frequency, self.trans_r, self.h_air, self.sig_iono, self.qwe_order
            )
        return self._ePrimary
