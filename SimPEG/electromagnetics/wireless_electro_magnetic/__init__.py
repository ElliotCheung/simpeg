""" module SimPEG.electromagnetics.wireless_electro_magnetic

SimPEG implementation of the WEM problem



"""

from .utils.analytic_1d import getEHfields
from ..natural_source.survey import Survey, Data
from ..natural_source.fields import Fields1DPrimarySecondary
from ..natural_source.simulation import Simulation1DPrimarySecondary
from . import sources
from . import receivers