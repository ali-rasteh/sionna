#
# SPDX-FileCopyrightText: Copyright (c) 2025 SHARP LABORATORIES OF AMERICA & NEW YORK UNIVERSITY (NYU). All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Suburban macrocell (SMa) channel model from 3GPP TR38.901 specification"""

from . import SystemLevelChannel
from . import SMaScenario

class SMa(SystemLevelChannel):
    # pylint: disable=line-too-long
    r"""
    Suburban macrocell (SMa) channel model from 3GPP [TR38901]_ specification.

    Setting up a SMa model requires configuring the network topology, i.e., the
    UTs and BSs locations, UTs velocities, etc. This is achieved using the
    :meth:`~sionna.phy.channel.tr38901.SMa.set_topology` method. Setting a different
    topology for each batch example is possible. The batch size used when setting up the network topology
    is used for the link simulations.

    The following code snippet shows how to setup an SMa channel model assuming
    an OFDM waveform:

    >>> # UT and BS panel arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                       num_cols_per_panel = 4,
    ...                       polarization = 'dual',
    ...                       polarization_type = 'cross',
    ...                       antenna_pattern = '38.901',
    ...                       carrier_frequency = 3.5e9)
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Instantiating SMa channel model
    >>> channel_model = SMa(carrier_frequency = 3.5e9,
    ...                     o2i_model = 'low',
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Setting up network topology
    >>> # ut_loc: UTs locations
    >>> # bs_loc: BSs locations
    >>> # ut_orientations: UTs array orientations
    >>> # bs_orientations: BSs array orientations
    >>> # in_state: Indoor/outdoor states of UTs
    >>> # residential_state: Residential states of UTs
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state,
    ...                            residential_state)
    >>> # Instanting the OFDM channel
    >>> channel = OFDMChannel(channel_model = channel_model,
    ...                       resource_grid = rg)

    where ``rg`` is an instance of :class:`~sionna.phy.ofdm.ResourceGrid`.

    Parameters
    -----------
    carrier_frequency : `float`
        Carrier frequency in Hertz

    o2i_model : "low" | "high" | "low-A" | "50/50" | "none"
        Outdoor to indoor (O2I) pathloss model, used for indoor UTs,
        see section 7.4.3 from 38.901 specification

    ut_array : :class:`~sionna.phy.channel.tr38901.PanelArray`
        Panel array configuration used by UTs

    bs_array : :class:`~sionna.phy.channel.tr38901.PanelArray`
        Panel array configuration used by BSs

    direction : "uplink" | "downlink"
        Link direction

    enable_pathloss : `bool`, (default `True`)
        If `True`, apply pathloss. Otherwise don't.

    enable_shadow_fading : `bool`, (default `True`)
        If `True`, apply shadow fading. Otherwise don't.

    always_generate_lsp : `bool`, (default `False`)
        If `True`, new large scale parameters (LSPs) are generated for every
        new generation of channel impulse responses. Otherwise, always reuse
        the same LSPs, except if the topology is changed.

    o2i_car_model : `None` (default) | "non-metalic"
        Outdoor to indoor (O2I) car pathloss model, used for outdoor UTs,
        see section 7.4.3 from 38.901 specification

    vegetation : `None` (default) | "no" | "sparse" | "dense"
        Vegetation density around the BSs. If `None`, it is set to "no".
        See section 7.4.2 of [TR38901]_ for details.

    calibration_mode : `bool`, (default `False`)
        If `True`, enable calibration mode. Default is `False`.
        
    near_field : `bool`, (default `False`)
        If `True`, use near-field approximation for the antenna arrays.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    num_time_steps : `int`
        Number of time steps

    sampling_frequency : `float`
        Sampling frequency [Hz]

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], `tf.complex`
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], `tf.float`
        Path delays [s]
    """
    def __init__(self, carrier_frequency, o2i_model, ut_array, bs_array,
        direction, enable_pathloss=True, enable_shadow_fading=True,
        always_generate_lsp=False, o2i_car_model=None, vegetation=None,
        calibration_mode=False, near_field=False, precision=None):

        # SMa scenario
        scenario = SMaScenario(carrier_frequency, o2i_model, ut_array, bs_array,
                               direction, enable_pathloss, enable_shadow_fading, 
                               o2i_car_model, vegetation, calibration_mode,
                               precision=precision)

        super().__init__(scenario, always_generate_lsp, near_field)
