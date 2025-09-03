# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
# pylint: disable=line-too-long, too-many-lines
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Utility functions for the channel module"""

from sionna.phy.block import Object


class ScenarioCalibrationParameters(Object):
    r"""
    Class for conveniently storing the calibration parameters of a scenario.

    Parameters
    -----------
    min_bs_ut_dist : `tf.float`
        Minimum BS-UT distance [m]

    isd : `tf.float`
        Inter-site distance [m]

    bs_height : `tf.float`
        BS elevation [m]

    min_ut_height : `tf.float`
        Minimum UT elevation [m]

    max_ut_height : `tf.float`
        Maximum UT elevation [m]

    indoor_probability : `tf.float`
        Probability of a UT to be indoor

    min_ut_velocity : `tf.float`
        Minimum UT velocity for outdoor UTs [m/s]

    max_ut_velocity : `tf.float`
        Maximim UT velocity for outdoor UTs [m/s]

    min_ut_velocity_indoor : `tf.float`, optional
        Minimum UT velocity for indoor UTs [m/s]. If not specified, the
        minimum outdoor UT velocity is used.

    max_ut_velocity_indoor : `tf.float`, optional
        Maximum UT velocity for indoor UTs [m/s]. If not specified, the
        maximum outdoor UT velocity is used.

    residential_probability : `tf.float`, optional
        Probability of a UT being in a residential area.
    """
    def __init__(self,  min_bs_ut_dist,
                        isd,
                        bs_height,
                        min_ut_height,
                        max_ut_height,
                        indoor_probability,
                        min_ut_velocity,
                        max_ut_velocity,
                        min_ut_velocity_indoor=None,
                        max_ut_velocity_indoor=None,
                        residential_probability=None):
        self.min_bs_ut_dist = min_bs_ut_dist
        self.isd = isd
        self.bs_height = bs_height
        self.min_ut_height = min_ut_height
        self.max_ut_height = max_ut_height
        self.indoor_probability = indoor_probability
        self.min_ut_velocity = min_ut_velocity
        self.max_ut_velocity = max_ut_velocity
        if min_ut_velocity_indoor is None:
            self.min_ut_velocity_indoor = min_ut_velocity
        else:
            self.min_ut_velocity_indoor = min_ut_velocity_indoor
        if max_ut_velocity_indoor is None:
            self.max_ut_velocity_indoor = max_ut_velocity
        else:
            self.max_ut_velocity_indoor = max_ut_velocity_indoor
        self.residential_probability = residential_probability
        super().__init__()


