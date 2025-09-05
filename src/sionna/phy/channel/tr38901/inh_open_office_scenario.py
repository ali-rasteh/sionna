#
# SPDX-FileCopyrightText: Copyright (c) 2025 SHARP LABORATORIES OF AMERICA & NEW YORK UNIVERSITY (NYU). All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""3GPP TR39.801 Indoor-open office channel model."""

import tensorflow as tf

from sionna.phy import SPEED_OF_LIGHT
from sionna.phy.utils import log10
from . import SystemLevelScenario
from . import ScenarioCalibrationParameters

class InHOpenOfficeScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 Indoor-open office channel model scenario

    Parameters
    -----------
    carrier_frequency : `float`
        Carrier frequency [Hz]

    o2i_model : "none"
        Outdoor to indoor (O2I) pathloss model, used for indoor UTs,
        see section 7.4.3 from 38.901 specification

    ut_array : :class:`~sionna.phy.channel.tr38901.PanelArray`
        Panel array configuration used by UTs

    bs_array : :class:`~sionna.phy.channel.tr38901.PanelArray`
        Panel array configuration used by BSs

    direction : "uplink" |"downlink"
        Link direction

    enable_pathloss : `bool`, (default `True`)
        If `True`, apply pathloss. Otherwise doesn't.

    enable_shadow_fading : `bool`, (default `True`)
        If `True`, apply shadow fading. Otherwise doesn't.

    o2i_car_model : `None`
        Outdoor to indoor (O2I) car pathloss model, used for outdoor UTs,
        see section 7.4.3.2 from 38.901 specification.

    release_number : "18" (default) | "19"
        Release number of the 3GPP specification to use.

    calibration_mode : `bool`, (default `False`)
        If `True`, enable calibration mode. Default is `False`.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    """

    #########################################
    # Public methods and properties
    #########################################

    def clip_carrier_frequency_lsp(self, fc):
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation

        Input
        -----
        fc : float
            Carrier frequency [GHz]

        Output
        -------
        : float
            Clipped carrier frequency, that should be used for LSp computation
        """
        if fc < 6.:
            fc = tf.cast(6., self.rdtype)
        return fc

    @property
    def min_2d_in(self):
        r"""Minimum indoor 2D distance for indoor UTs [m],
        Never used in this scenario, but required by the base class."""
        return tf.constant(0.0, self.rdtype)

    @property
    def max_2d_in(self):
        r"""Maximum indoor 2D distance for indoor UTs [m],
        Never used in this scenario, but required by the base class."""
        return tf.constant(0.0, self.rdtype)

    @property
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of UTs.

        Computed following section 7.4.2 of TR 38.901.

        [batch size, num_ut]"""

        distance_2d_in = self._distance_2d_in
        
        cond1 = distance_2d_in <= 5.0
        cond2 = tf.logical_and(distance_2d_in > 5.0, distance_2d_in <= 49.0)

        los_probability_1 = tf.ones_like(distance_2d_in)
        los_probability_2 = tf.exp(-(distance_2d_in-5.)/70.8)
        los_probability_3 = tf.exp(-(distance_2d_in-49.)/211.7) * 0.54

        los_probability = tf.where(cond1, los_probability_1,
                    tf.where(cond2, los_probability_2, los_probability_3))
        
        return los_probability

    @property
    def rays_per_cluster(self):
        r"""Number of rays per cluster"""
        return tf.constant(20, tf.int32)

    @property
    def s_trp_parameters(self):
        r"""Tuple containing the parameters for the Near-Field (NF) S_TRP model
        
        (K1, Alpha, Beta)
        
        K1 : `int`
            K1 parameter
        Alpha : `tf.float`
            Alpha parameter
        Beta : `tf.float`
            Beta parameter
        """
        return (4,
                tf.constant(1.25, self.rdtype),
                tf.constant(1.27, self.rdtype))

    @property
    def los_parameter_filepath(self):
        r""" Path of the configuration file for LoS scenario"""
        return 'InHOpenOffice_LoS.json'

    @property
    def nlos_parameter_filepath(self):
        r""" Path of the configuration file for NLoS scenario"""
        return'InHOpenOffice_NLoS.json'

    @property
    def o2i_parameter_filepath(self):
        r""" Path of the configuration file for indoor scenario"""
        return 'InHOpenOffice_O2I.json'

    #########################
    # Utility methods
    #########################

    def _compute_lsp_log_mean_std(self):
        r"""Computes the mean and standard deviations of LSPs in log-domain"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        h_bs = self.h_bs
        h_bs = tf.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = tf.expand_dims(h_ut, axis=1) # For broadcasting
        fc = self.carrier_frequency

        ## Mean
        # DS
        log_mean_ds = self.get_param("muDS")
        # ASD
        log_mean_asd = self.get_param("muASD")
        # ASA
        log_mean_asa = self.get_param("muASA")
        # SF.  Has zero-mean.
        log_mean_sf = tf.zeros([batch_size, num_bs, num_ut],
                                self.rdtype)
        # K.  Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK")/10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD
        log_mean_zsd_los = -1.43 * log10(1.+fc/1e9) + 2.228
        log_mean_zsd_nlos = tf.constant(1.08, self.rdtype),
        log_mean_zsd = tf.where(self.los, log_mean_zsd_los, log_mean_zsd_nlos)
        # Excess delay for absolute time of arrival (ToA) estimation
        log_mean_ed_nlos = tf.constant(-8.6, self.rdtype)
        # A very small value for LoS case
        log_mean_ed_los = tf.constant(-30.0, self.rdtype)
        log_mean_ed = tf.where(self.los, log_mean_ed_los, log_mean_ed_nlos)

        lsp_log_mean = tf.stack([log_mean_ds,
                                log_mean_asd,
                                log_mean_asa,
                                log_mean_sf,
                                log_mean_k,
                                log_mean_zsa,
                                log_mean_zsd,
                                log_mean_ed], axis=3)

        ## STD
        # DS
        log_std_ds = self.get_param("sigmaDS")
        # ASD
        log_std_asd = self.get_param("sigmaASD")
        # ASA
        log_std_asa = self.get_param("sigmaASA")
        # SF. Given in dB in the 3GPP tables, hence the division by 10
        # O2I and NLoS cases just require the use of a predefined value
        log_std_sf = self.get_param("sigmaSF")/10.0
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK")/10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        # log_std_zsd = self.get_param("sigmaZSD")
        log_std_zsd_los = 0.13 * log10(1+fc/1e9) + 0.30
        log_std_zsd_nlos = tf.constant(0.36, self.rdtype)
        log_std_zsd = tf.where(self.los, log_std_zsd_los, log_std_zsd_nlos)
        # Excess delay for absolute time of arrival (ToA) estimation
        log_std_ed_nlos = tf.constant(0.1, self.rdtype)
        log_std_ed_los = tf.constant(0., self.rdtype)
        log_std_ed = tf.where(self.los, log_std_ed_los, log_std_ed_nlos)

        lsp_log_std = tf.stack([log_std_ds,
                               log_std_asd,
                               log_std_asa,
                               log_std_sf,
                               log_std_k,
                               log_std_zsa,
                               log_std_zsd,
                               log_std_ed], axis=3)

        self._lsp_log_mean = lsp_log_mean
        self._lsp_log_std = lsp_log_std

        # ZOD offset
        zod_offset_los = tf.constant(0.0, self.rdtype)
        zod_offset_nlos = tf.constant(0.0, self.rdtype)
        zod_offset = tf.where(self.los, zod_offset_los, zod_offset_nlos)
        self._zod_offset = zod_offset

    def _compute_pathloss_basic(self):
        r"""Computes the basic component of the pathloss [dB]"""

        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        fc = self.carrier_frequency # Carrier frequency (Hz)
        h_bs = self.h_bs
        h_bs = tf.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = tf.expand_dims(h_ut, axis=1) # For broadcasting

        ## Basic path loss for LoS
        pl_los = 32.4 + 17.3*log10(distance_3d) + 20.0*log10(fc/1e9)

        ## Basic pathloss for NLoS and O2I
        pl_1 = 38.3*log10(distance_3d) + 17.3 + 24.9*log10(fc/1e9)
        pl_nlos = tf.math.maximum(pl_los, pl_1)

        ## Set the basic pathloss according to UT state

        # Expand to allow broadcasting with the BS dimension
        # LoS
        pl_b = tf.where(self.los, pl_los, pl_nlos)

        self._pl_b = pl_b

    def get_calibration_parameters(self, nearfield=False):
        r"""Returns the calibration parameters for the InHOpenOffice scenario

        Input
        -----
        nearfield : `bool`, (default `False`)
            If `True`, returns the calibration parameters for the near-field
            regime. Otherwise, returns the calibration parameters for the far-field
            regime.

        Output
        -------
        : `dict`
            Dictionary containing the calibration parameters
        """
        if nearfield:
            parameters = ScenarioCalibrationParameters(
                min_bs_ut_dist = tf.constant(0., self.rdtype),
                isd = tf.constant(20., self.rdtype),
                bs_height = tf.constant(3., self.rdtype),
                min_ut_height = tf.constant(1., self.rdtype),
                max_ut_height = tf.constant(1., self.rdtype),
                # indoor probability is 0 to prevent the o2i from being applied
                indoor_probability = tf.constant(0.0, self.rdtype),
                min_ut_velocity = tf.constant(3./3.6, self.rdtype),
                max_ut_velocity = tf.constant(3./3.6, self.rdtype)
            )
        else:
            parameters = ScenarioCalibrationParameters(
                min_bs_ut_dist = tf.constant(0., self.rdtype),
                isd = tf.constant(20., self.rdtype),
                bs_height = tf.constant(3., self.rdtype),
                min_ut_height = tf.constant(1., self.rdtype),
                max_ut_height = tf.constant(1., self.rdtype),
                # indoor probability is 0 to prevent the o2i from being applied
                indoor_probability = tf.constant(0.0, self.rdtype),
                min_ut_velocity = tf.constant(3./3.6, self.rdtype),
                max_ut_velocity = tf.constant(3./3.6, self.rdtype)
            )
        return parameters
    
    def _sample_indoor_distance(self):
        r"""Set indoor distances equal to the total distances and
        outdoor distances to zero, because this scenario
        does not have any outdoor UTs.
        """
        # Sample the indoor 2D distances for each BS-UT link
        self._distance_2d_in = self.distance_2d
        # Compute the outdoor 2D distances
        self._distance_2d_out = tf.zeros_like(self.distance_2d, dtype=self.rdtype)
        # Compute the indoor 3D distances
        self._distance_3d_in = self.distance_3d
        # Compute the outdoor 3D distances
        self._distance_3d_out = tf.zeros_like(self.distance_3d, dtype=self.rdtype)



