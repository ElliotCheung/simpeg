from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import TYPE_CHECKING

from discretize.base import BaseMesh
from SimPEG import maps
from .base import (
    RegularizationMesh,
    BaseRegularization
)
from .sparse import (
    Sparse,
    SparseSmall,
    SparseDeriv
)
from .. import utils

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


class BaseVectorAmplitude(BaseRegularization):
    """
    Base vector amplitude function.
    """
    _projection = None

    @property
    def projection(self):
        """Projection matrix from vector components to amplitude."""
        if getattr(self, "_projection", None) is None:
            self._projection = sp.hstack([
                sp.identity(self.nP),
                sp.identity(self.nP),
                sp.identity(self.nP)
            ])

        return self._projection

    def amplitude_map(self, m):
        """Create sparse vector model."""
        return self.projection @ utils.sdiag(m)

    @property
    def mapping(self) -> maps.IdentityMap:
        """Mapping applied to the model values"""
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: maps.IdentityMap):
        if mapping is None:
            mapping = maps.IdentityMap()
        if not isinstance(mapping, maps.IdentityMap):
            raise TypeError(
                f"'mapping' must be of type {maps.IdentityMap}. "
                f"Value of type {type(mapping)} provided."
            )
        self._mapping = mapping


class VectorAmplitudeSmall(SparseSmall, BaseVectorAmplitude):
    """
    Sparse smallness regularization on vector amplitude.

    **Inputs**

    :param int norm: norm on the smallness
    """

    def f_m(self, m):
        """
        Compute the amplitude of a vector model.
        """
        m = self.amplitude_map(m) * m
        return self.mapping * self._delta_m(m)

    def f_m_deriv(self, m) -> csr_matrix:
        m = self.amplitude_map(m) * m

        return self.mapping.deriv(self._delta_m(m)) @ self.projection


class VectorAmplitudeDeriv(SparseDeriv, BaseVectorAmplitude):
    """
    Base Class for sparse regularization on first spatial derivatives
    """

    def f_m(self, m):
        m = self.amplitude_map(m) * m
        dfm_dl = self.cell_gradient @ (self._delta_m(m))

        return dfm_dl

    def f_m_deriv(self, m) -> csr_matrix:
        m = self.amplitude_map(m) * m
        return (self.cell_gradient @ self.mapping.deriv(self._delta_m(m))) @ self.projection


class VectorAmplitude(Sparse):
    """
    The regularization is:

    .. math::

        R(m) = \\frac{1}{2}\\mathbf{(m-m_\\text{ref})^\\top W^\\top R^\\top R
        W(m-m_\\text{ref})}

    where the IRLS weight

    .. math::

        R = \\eta TO FINISH LATER!!!

    So the derivative is straight forward:

    .. math::

        R(m) = \\mathbf{W^\\top R^\\top R W (m-m_\\text{ref})}

    The IRLS weights are recomputed after each beta solves.
    It is strongly recommended to do a few Gauss-Newton iterations
    before updating.
    """

    def __init__(
        self,
        mesh,
        active_cells=None,
        **kwargs,
    ):
        if not isinstance(mesh, RegularizationMesh):
            mesh = RegularizationMesh(mesh)

        if not isinstance(mesh, RegularizationMesh):
            TypeError(
                f"'regularization_mesh' must be of type {RegularizationMesh} or {BaseMesh}. "
                f"Value of type {type(mesh)} provided."
            )
        self._regularization_mesh = mesh

        if active_cells is not None:
            self._regularization_mesh.active_cells = active_cells

        objfcts = [
            VectorAmplitudeSmall(mesh=self.regularization_mesh),
            VectorAmplitudeDeriv(mesh=self.regularization_mesh, orientation="x"),
        ]

        if mesh.dim > 1:
            objfcts.append(VectorAmplitudeDeriv(mesh=self.regularization_mesh, orientation="y"))

        if mesh.dim > 2:
            objfcts.append(VectorAmplitudeDeriv(mesh=self.regularization_mesh, orientation="z"))

        super().__init__(
            self.regularization_mesh,
            objfcts=objfcts,
            **kwargs,
        )

