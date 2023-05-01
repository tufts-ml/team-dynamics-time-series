from dataclasses import dataclass

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.params import AllParameters_JAX
from dynagroup.types import NumpyArray2D, NumpyArray3D


@dataclass
class InitializationResults:
    params: AllParameters_JAX
    ES_summary: HMM_Posterior_Summary_JAX
    EZ_summaries: HMM_Posterior_Summaries_JAX
    record_of_most_likely_system_states: NumpyArray2D  # Txnum_EM_iterations
    record_of_most_likely_entity_states: NumpyArray3D  # TxJx num_EM_iterations
