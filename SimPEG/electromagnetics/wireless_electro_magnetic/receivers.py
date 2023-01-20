from ..natural_source.receivers import PointNaturalSource
from ...utils.code_utils import deprecate_class, validate_string

import numpy as np
from scipy.constants import mu_0

def _alpha(src):
    return 1 / (2 * np.pi * mu_0 * src.frequency)

#################
#   Receivers   #
#################

class PointWEM(PointNaturalSource):
    """
    Point receiver class for WEM simulation.

    Assumes that the data locations are standard xyz coordinates;
    i.e. (x,y,z) is (Easting, Northing, up).

    Parameters
    ----------
    locations : (n_loc, n_dim) numpy.ndarray
        Receiver locations.
    orientation : {'xx', 'xy', 'yx', 'yy'}
        MT receiver orientation.
    component : {'real', 'imag', 'apparent_resistivity', 'phase'}
        MT data type.
    """

    @property
    def component(self):
        """Data type; i.e. "real", "imag", "apparent_resistivity", "phase", "EF_R", "HF_I"

        Returns
        -------
        str
            Data type; i.e. "real", "imag", "apparent_resistivity", "phase", "EF_R", "HF_I"
        """
        return self._component

    @component.setter
    def component(self, var):
        self._component = validate_string(
            "component",
            var,
            [
                ("real", "re", "in-phase", "in phase"),
                ("imag", "imaginary", "im", "out-of-phase", "out of phase"),
                (
                    "apparent_resistivity",
                    "apparent resistivity",
                    "appresistivity",
                    "apparentresistivity",
                    "apparent-resistivity",
                    "apparent_resistivity",
                    "appres",
                    "app_res",
                    "rho",
                    "rhoa",
                ),
                ("phase", "phi"),
                ("Efield real", "EF_R"),
                ("Efield imag", "EF_I"),
                ("Hfield real", "HF_R"),
                ("Hfield imag", "HF_I"),
            ],
        )

    def eval(self, src, mesh, f, return_complex=False):
        """
        Project the fields to WEM data.

        Parameters
        ----------
        src : SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc
            WEM source
        mesh : discretize.TensorMesh mesh
            Mesh on which the discretize solution is obtained
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            WEM fields object of the source
        return_complex : bool (optional)
            Flag for return the complex evaluation

        Returns
        -------
        numpy.ndarray
            Evaluated data for the receiver
        """
        if self.component.split()[0] == "Efield":
            e_arr = f[src, "e"]
            if mesh.dim == 1:
                e_loc = f.aliasFields["e"][1]
                PE = self.getP(mesh, e_loc)
                e = PE @ e_arr[:, 0]
                return getattr(e, self.component.split()[1])

        if self.component.split()[0] == "Hfield":
            h_arr = f[src, "h"]
            if mesh.dim == 1:
                h_loc = f.aliasFields["h"][1]
                PH = self.getP(mesh, h_loc)
                h = PH @ h_arr[:, 0]
                return getattr(h, self.component.split()[1])

        imp = self._eval_impedance(src, mesh, f)
        if return_complex:
            return imp
        elif self.component == "apparent_resistivity":
            return _alpha(src) * (imp.real ** 2 + imp.imag ** 2)
        elif self.component == "phase":
            return 180 / np.pi * (np.arctan2(imp.imag, imp.real))
        else:
            return getattr(imp, self.component)

    def evalDeriv(self, src, mesh, f, du_dm_v=None, v=None, adjoint=False):
        """Derivative of projection with respect to the fields

        Parameters
        ----------
        str : SimPEG.electromagnetics.frequency_domain.sources.BaseFDEMSrc
            WEM source
        mesh : discretize.TensorMesh
            Mesh on which the discretize solution is obtained
        f : SimPEG.electromagnetics.frequency_domain.fields.FieldsFDEM
            WEM fields object of the source
        du_dm_v : None,
            Supply pre-computed derivative?
        v : numpy.ndarray
            Vector of size
        adjoint : bool, default = ``False``
            If ``True``, compute the adjoint operation

        Returns
        -------
        numpy.ndarray
            Calculated derivative (nD,) (adjoint=False) and (nP,2) (adjoint=True) for both polarizations
        """
        if self.component.split()[0] == "Efield":
            if mesh.dim == 1:
                e_loc = f.aliasFields["e"][1]
                PE = self.getP(mesh, e_loc)
                if adjoint:
                    v = (1+0j)*v if (self.component.split()[1] == "real") else -1j*v
                    return f._eDeriv(src, None, PE.T @ v, adjoint=True)
                eDeriv = PE @ f._eDeriv(src, du_dm_v, v, adjoint=False)
                return getattr(eDeriv, self.component.split()[1])

        if self.component.split()[0] == "Hfield":
            if mesh.dim == 1:
                h_loc = f.aliasFields["h"][1]
                PH = self.getP(mesh, h_loc)
                if adjoint:
                    v = -1j*v if (self.component.split()[1] == "real") else (1+0j)*v
                    return f._hDeriv(src, None, PH.T @ v, adjoint=True)
                hDeriv = PH @ f._hDeriv(src, du_dm_v, v, adjoint=False)
                return getattr(hDeriv, self.component.split()[1])
        
        return self._eval_impedance_deriv(
            src, mesh, f, du_dm_v=du_dm_v, v=v, adjoint=adjoint
        )
    pass