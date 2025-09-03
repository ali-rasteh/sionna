#
# SPDX-FileCopyrightText: Copyright (c) 2025 SHARP CORPORATION & NEW YORK UNIVERSITY(NYU). All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""3GPP TR39.801 suburban macrocell (SMa) channel model"""

import tensorflow as tf

from sionna.phy import SPEED_OF_LIGHT, PI, config
from sionna.phy.utils import log10
from . import SystemLevelScenario
from . import ScenarioCalibrationParameters

class SMaScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.901 suburban macrocell (SMa) channel model scenario

    Parameters
    -----------
    carrier_frequency : `float`
        Carrier frequency [Hz]

    o2i_model : "low" | "high" | "low-A" | "50/50"
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

    o2i_car_model : `None` (default) | "non-metalic"
        Outdoor to indoor (O2I) car pathloss model, used for outdoor UTs,
        see section 7.4.3 from 38.901 specification

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    """
    def __init__(self, carrier_frequency, o2i_model, ut_array, bs_array,
        direction, enable_pathloss=True, enable_shadow_fading=True,
        o2i_car_model=None, vegetation=None, calibration_mode=False,
        precision=None):

        assert carrier_frequency > 0.5e9 and carrier_frequency < 37e9, \
            "SMa scenario is only defined for carrier frequencies > 0.5 GHz and < 37 GHz"
        
        assert o2i_model in ('low', 'high', 'low-A', '50/50'), \
            "o2i_model must be 'low', 'high', 'low-A' or '50/50'"

        # SMa is only defined for release 19
        super().__init__(carrier_frequency, o2i_model, ut_array, bs_array,
            direction, enable_pathloss, enable_shadow_fading,
            o2i_car_model, release_number="19", calibration_mode=calibration_mode,
            precision = precision)

        assert (vegetation is None) or (vegetation in ["no", "sparse", "dense"]), \
            "vegetation must be None, 'no', 'sparse' or 'dense'."

        if vegetation is None:
            vegetation = "no"
        self._vegetation = vegetation

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
        return fc

    @property
    def min_2d_in(self):
        r"""Minimum indoor 2D distance for indoor UTs [m]
        [batch size, num_bs, num_ut]"""
        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        min_2d_in = tf.zeros([batch_size, num_bs, num_ut], self.rdtype)
        return min_2d_in

    @property
    def max_2d_in(self):
        r"""Maximum indoor 2D distance for indoor UTs [m]
        [batch size, num_bs, num_ut]"""
        num_bs = self.num_bs
        indoor = self.indoor
        indoor_mask = tf.where(indoor, tf.constant(1.0, self.rdtype),
            tf.constant(0.0, self.rdtype))
        max_2d_in = tf.where(self._residential_state, tf.constant(10.0, self.rdtype),
            tf.constant(25.0, self.rdtype)) * indoor_mask
        max_2d_in = tf.expand_dims(max_2d_in, axis=1)
        max_2d_in = tf.tile(max_2d_in, [1, num_bs, 1])
        return max_2d_in

    @property
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs.

        Computed following section 7.4.2 of TR 38.901.

        [batch size, num_ut]"""

        h_bs = self.h_bs
        h_bs = tf.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = tf.expand_dims(h_ut, axis=1) # For broadcasting

        distance_2d_out = self._distance_2d_out

        d_clutter = tf.constant(30.0, self.rdtype)
        h_commercial = tf.constant(20.0, self.rdtype)
        h_residential = tf.constant(8.0, self.rdtype)
        h_vegetation = tf.constant(15.0, self.rdtype)
        r_commercial = tf.constant(0.02, self.rdtype)
        r_residential = tf.constant(0.18, self.rdtype)
        if self._vegetation == 'no':
            r_vegetation = tf.constant(0.0, self.rdtype)
        elif self._vegetation == 'sparse':
            r_vegetation = tf.constant(0.1, self.rdtype)
        elif self._vegetation == 'dense':
            r_vegetation = tf.constant(0.2, self.rdtype)

        k_commercial = -(d_clutter/(tf.math.log(1.0-r_commercial))) \
            * (h_bs-h_ut)/(h_commercial-h_ut)
        k_residential = -(d_clutter/(tf.math.log(1.0-r_residential))) \
            * (h_bs-h_ut)/tf.math.maximum((h_residential-h_ut), tf.constant(0.0, self.rdtype))
        k_vegetation = -(d_clutter/(tf.math.log(1.0-r_vegetation))) \
            * (h_bs-h_ut)/(h_vegetation-h_ut)

        los_probability = tf.math.exp(-distance_2d_out/k_commercial) \
        * tf.math.exp(-distance_2d_out/k_residential) \
        * tf.math.exp(-distance_2d_out/k_vegetation)

        los_probability = tf.where(tf.math.less(distance_2d_out, 10.0),
            tf.constant(1.0, self.rdtype), los_probability)
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
        return (0,
                tf.constant(0., self.rdtype),
                tf.constant(0., self.rdtype))

    @property
    def los_parameter_filepath(self):
        r""" Path of the configuration file for LoS scenario"""
        return 'SMa_LoS_rel19.json'

    @property
    def nlos_parameter_filepath(self):
        r""" Path of the configuration file for NLoS scenario"""
        return 'SMa_NLoS_rel19.json'

    @property
    def o2i_parameter_filepath(self):
        r""" Path of the configuration file for indoor scenario"""
        return 'SMa_O2I_rel19.json'

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
        log_mean_zsd = self.get_param("muZSD")
        # Excess delay for absolute time of arrival (ToA) estimation
        log_mean_ed_nlos = tf.constant(-7.702, self.rdtype)
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
        log_std_sf_o2i_nlos = self.get_param("sigmaSF")/10.0
        # For LoS, two possible scenarion depending on the 2D location of the
        # user
        distance_breakpoint = (2.*PI*h_bs*h_ut*self.carrier_frequency/
            SPEED_OF_LIGHT)
        log_std_sf_los=tf.where(tf.math.less(distance_2d, distance_breakpoint),
            self.get_param("sigmaSF1")/10.0, self.get_param("sigmaSF2")/10.0)
        # Use the correct SF STD according to the user scenario: NLoS/O2I, or
        # LoS
        log_std_sf = tf.where(self.los, log_std_sf_los, log_std_sf_o2i_nlos)
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK")/10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        log_std_zsd = self.get_param("sigmaZSD")
        # Excess delay for absolute time of arrival (ToA) estimation
        log_std_ed_nlos = tf.constant(0.4, self.rdtype)
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
        h_cl = 3.5
        zod_offset = (tf.atan((h_bs-h_cl)/distance_2d)
          - tf.atan((h_bs-h_ut)/distance_2d))
        zod_offset = tf.where(tf.math.less(h_ut, h_cl),
            zod_offset, tf.constant(0., self.rdtype))
        zod_offset = tf.where(self.los,
                            tf.constant(0., self.rdtype), zod_offset)
        self._zod_offset = zod_offset

    def _compute_pathloss_basic(self):
        r"""Computes the basic component of the pathloss [dB]"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        fc = self.carrier_frequency # Carrier frequency (Hz)
        h_bs = self.h_bs
        h_bs = tf.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = tf.expand_dims(h_ut, axis=1) # For broadcasting

        ## Basic path loss for LoS

        distance_breakpoint = (2.*PI*h_bs*h_ut*fc/SPEED_OF_LIGHT)
        h = tf.constant(10., self.rdtype)
        pl_1 = (20.0*log10(40.0*PI*distance_3d*fc/1e9/3.)
            + tf.math.minimum(0.03*tf.math.pow(h,1.72),
                10.0)*log10(distance_3d)
            - tf.math.minimum(0.044*tf.math.pow(h,1.72), 14.77)
            + 0.002*log10(h)*distance_3d)
        pl_2 = (20.0*log10(40.0*PI*distance_breakpoint*fc/1e9/3.)
            + tf.math.minimum(0.03*tf.math.pow(h,1.72),
                10.0)*log10(distance_breakpoint)
            - tf.math.minimum(0.044*tf.math.pow(h,1.72), 14.77)
            + 0.002*log10(h)*distance_breakpoint
            + 40.0*log10(distance_3d/distance_breakpoint))

        pl_los = tf.where(tf.math.less(distance_2d, distance_breakpoint),
            pl_1, pl_2)

        ## Basic pathloss for NLoS and O2I

        h = tf.constant(10., self.rdtype)
        W = tf.constant(10., self.rdtype)
        pl_nlos = (161.04 - 7.1*log10(W) + 7.5*log10(h)
                - (24.37 - 3.7*tf.square(h/h_bs)) * log10(h_bs)
                + (43.42 - 3.1*log10(h_bs)) * (log10(distance_3d)-3.0)
                + 20.0*log10(fc/1e9) - (3.2*tf.square(log10(11.75*h_ut))
                - 4.97))

        ## Set the basic pathloss according to UT state
        pl_b = tf.where(self.los, pl_los, pl_nlos)

        self._pl_b = pl_b

    def get_calibration_parameters(self, nearfield=False):
        r"""Returns the calibration parameters for the SMa scenario

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
                min_bs_ut_dist = tf.constant(35., self.rdtype),
                isd = tf.constant(1299., self.rdtype),
                bs_height = tf.constant(35., self.rdtype),
                min_ut_height = tf.constant(1.5, self.rdtype),
                max_ut_height = tf.constant(1.5, self.rdtype),
                indoor_probability = tf.constant(0.8, self.rdtype),
                min_ut_velocity = tf.constant(40./3.6, self.rdtype),
                max_ut_velocity = tf.constant(40./3.6, self.rdtype),
                min_ut_velocity_indoor = tf.constant(3./3.6, self.rdtype),
                max_ut_velocity_indoor = tf.constant(3./3.6, self.rdtype),
                residential_probability = tf.constant(0.9, self.rdtype)
            )
        else:
            parameters = ScenarioCalibrationParameters(
                min_bs_ut_dist = tf.constant(35., self.rdtype),
                isd = tf.constant(1299., self.rdtype),
                bs_height = tf.constant(35., self.rdtype),
                min_ut_height = tf.constant(1.5, self.rdtype),
                max_ut_height = tf.constant(1.5, self.rdtype),
                indoor_probability = tf.constant(0.8, self.rdtype),
                min_ut_velocity = tf.constant(40./3.6, self.rdtype),
                max_ut_velocity = tf.constant(40./3.6, self.rdtype),
                min_ut_velocity_indoor = tf.constant(3./3.6, self.rdtype),
                max_ut_velocity_indoor = tf.constant(3./3.6, self.rdtype),
                residential_probability = tf.constant(0.9, self.rdtype)
            )
        return parameters
