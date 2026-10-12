import numpy as np

from oceantracker.trajectory_modifiers._base_trajectory_modifers import (
    _BaseTrajectoryModifier
)
from oceantracker.util.parameter_checking import ParamValueChecker as PVC
from oceantracker.util.numba_util import njitOT
from oceantracker.shared_info import shared_info as si


class DNADecay(_BaseTrajectoryModifier):
    """
    Stochastic degradation of DNA copies carried by particles.

    Each DNA copy survives a time step with probability:

        p = exp(-dt / decay_time_scale)

    The number of copies is updated using a binomial distribution.
    Particles are marked as dead when their DNA copy count reaches zero.
    """

    def __init__(self):
        super().__init__()

        self.add_default_params(
            decay_time_scale=PVC(
                86400.0,
                float,
                min=1.0e-12,
                doc_str=(
                    "Characteristic DNA decay time scale in seconds."
                ),
                units="sec"
            )
        )

    def check_requirements(self):
        self.check_class_required_fields_prop_etc(
            required_props_list=["dna_copies", "status"]
        )

    def update(self, n_time_step, time_sec, active):

        part_prop = si.class_roles.particle_properties

        dna_copies = part_prop["dna_copies"].data
        status = part_prop["status"].data

        decay_time_scale = self.params["decay_time_scale"]

        # Use the absolute time step so that the probability remains
        # valid even if the simulation uses a negative time step.
        dt = abs(si.settings.time_step)

        # Probability that an individual DNA copy survives this step.
        p_survival = np.exp(-dt / decay_time_scale)

        self._decay_and_cull(
            dna_copies,
            status,
            p_survival,
            si.particle_status_flags.dead,
            active
        )

    @staticmethod
    @njitOT
    def _decay_and_cull(
        dna_copies,
        status,
        p_survival,
        status_dead,
        active
    ):

        for i in active:

            # Ignore particles that have already been marked as dead.
            if status[i] < 0:
                continue

            # A particle with no DNA copies must be dead.
            if dna_copies[i] <= 0:
                dna_copies[i] = 0
                status[i] = status_dead
                continue

            # np.random.binomial needs an int number, otherwise gives an error
            n_copies = int(dna_copies[i])

            # Stochastic degradation of the DNA copies.
            remaining = np.random.binomial(
                n_copies,
                p_survival
            )

            dna_copies[i] = remaining

            # Kill the particle if no copies remain.
            if remaining == 0:
                status[i] = status_dead