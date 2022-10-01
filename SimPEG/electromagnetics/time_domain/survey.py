# import properties
from ...survey import BaseSurvey
from .sources import BaseTDEMSrc
import numpy as np


####################################################
# Survey
####################################################


class Survey(BaseSurvey):
    """Time domain electromagnetic survey

    Parameters
    ----------
    source_list : list of SimPEG.electromagnetic.time_domain.sources.BaseTDEMSrc
        List of SimPEG TDEM sources
    """

    # source_list = properties.List(
    #     "A list of sources for the survey",
    #     properties.Instance("A SimPEG source", BaseTDEMSrc),
    #     default=[],
    # )

    def __init__(self, source_list=None, **kwargs):
        if source_list is None:
            raise AttributeError("Survey cannot be instantiated without sources")
        super(Survey, self).__init__(source_list, **kwargs)

    @property
    def source_list(self):
        """List of TDEM sources associated with the survey

        Returns
        -------
        list of BaseTDEMSrc
            List of TDEM sources associated with the survey
        """
        return self._source_list

    @source_list.setter
    def source_list(self, new_list):
        if not isinstance(new_list, list):
            new_list = [new_list]

        if any([isinstance(x, BaseTDEMSrc) == False for x in new_list]):
            raise TypeError("Source list must be a list of SimPEG.survey.BaseTDEMSrc")

        assert len(set(new_list)) == len(
            new_list
        ), "The source_list must be unique. Cannot re-use sources"

        self._sourceOrder = dict()
        # [self._sourceOrder.setdefault(src._uid, ii) for ii, src in enumerate(new_list)]
        ii = 0
        for src in new_list:
            n_fields = src._fields_per_source
            self._sourceOrder[src._uid] = [ii + i for i in range(n_fields)]
            ii += n_fields
        self._source_list = new_list
