#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Urban microcell (UMi) channel model from 3GPP TR38.901 specification"""

from . import SystemLevelChannel
from . import UMiScenario

class UMi(SystemLevelChannel):
    # pylint: disable=line-too-long
    r"""
    Urban microcell (UMi) channel model from 3GPP [TR38901]_ specification

    Setting up a UMi model requires configuring the network topology, i.e., the
    UTs and BSs locations, UTs velocities, etc. This is achieved using the
    :meth:`~sionna.phy.channel.tr38901.UMi.set_topology` method. Setting a different
    topology for each batch example is possible. The batch size used when setting up the network topology
    is used for the link simulations.

    The following code snippet shows how to setup a UMi channel model operating
    in the frequency domain:

    >>> # UT and BS panel arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                       num_cols_per_panel = 4,
    ...                       polarization = 'dual',
    ...                       polarization_type  = 'cross',
    ...                       antenna_pattern = '38.901',
    ...                       carrier_frequency = 3.5e9)
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Instantiating UMi channel model
    >>> channel_model = UMi(carrier_frequency = 3.5e9,
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
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> # Instanting the frequency domain channel
    >>> channel = OFDMChannel(channel_model = channel_model,
    ...                       resource_grid = rg)

    where ``rg`` is an instance of :class:`~sionna.phy.ofdm.ResourceGrid`.

    Parameters
    -----------
    carrier_frequency : `float`
        Carrier frequency in Hertz

    o2i_model : "low" | "high"
        Outdoor-to-indoor loss model for UTs located indoor.
        Set this parameter to "low" to use the low-loss model, or to "high"
        to use the high-loss model.
        See section 7.4.3 of [TR38901]_ for details.

    rx_array : :class:`~sionna.phy.channel.tr38901.PanelArray`
        Panel array used by the receivers. All receivers share the same
        antenna array configuration.

    tx_array : :class:`~sionna.phy.channel.tr38901.PanelArray`
        Panel array used by the transmitters. All transmitters share the
        same antenna array configuration.

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
        always_generate_lsp=False, release_number="19.0.0", precision=None):

        # RMa scenario
        scenario = UMiScenario(carrier_frequency, o2i_model, ut_array, bs_array,
                               direction, enable_pathloss, enable_shadow_fading, release_number,
                               precision)

        super().__init__(scenario, always_generate_lsp)
