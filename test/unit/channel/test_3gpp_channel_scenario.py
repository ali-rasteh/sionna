#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import tensorflow as tf
import unittest
import numpy as np
from sionna.phy import channel
from channel_test_utils import *


class TestScenario(unittest.TestCase):
    r"""Test the distance calculations and function that get the parameters
    according to the scenario.
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 100

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR = 1e-2

    # Height of UTs
    H_UT = 1.5

    # Height of BSs
    H_BS = 10.0

    # Number of BS
    NB_BS = 3

    # Number of UT
    NB_UT = 10

    def setUp(self):

        batch_size = TestScenario.BATCH_SIZE
        nb_bs = TestScenario.NB_BS
        nb_ut = TestScenario.NB_UT
        fc = TestScenario.CARRIER_FREQUENCY
        h_ut = TestScenario.H_UT
        h_bs = TestScenario.H_BS

        # UT and BS arrays have no impact on LSP
        # However, these are needed to instantiate the model
        self.bs_array = channel.tr38901.PanelArray(num_rows_per_panel=2,
                                              num_cols_per_panel=2,
                                              polarization='dual',
                                              polarization_type='VH',
                                              antenna_pattern='38.901',
                                              carrier_frequency=fc)
        self.ut_array = channel.tr38901.PanelArray(num_rows_per_panel=1,
                                              num_cols_per_panel=1,
                                              polarization='dual',
                                              polarization_type='VH',
                                              antenna_pattern='38.901',
                                              carrier_frequency=fc)

        # The following quantities have no impact on LSP
        # However,these are needed to instantiate the model
        ut_orientations = tf.zeros([batch_size, nb_ut])
        bs_orientations = tf.zeros([batch_size, nb_ut])
        ut_velocities = tf.zeros([batch_size, nb_ut])

        self.scenario = channel.tr38901.RMaScenario(fc, self.ut_array,
                                                    self.bs_array, "uplink")

        ut_loc = generate_random_loc(batch_size, nb_ut, (100,2000),
                                     (100,2000), (h_ut, h_ut))
        bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                            (0,100), (h_bs, h_bs))

        in_state = generate_random_bool(batch_size, nb_ut, 0.5)
        self.scenario.set_topology(ut_loc, bs_loc, ut_orientations,
                                bs_orientations, ut_velocities, in_state)

    def test_dist(self):
        """Test calculation of distances (total, in, and out)"""
        d_3d = self.scenario.distance_3d
        d_3d_in = self.scenario.distance_3d_in
        d_3d_out = self.scenario.distance_3d_out
        d_2d = self.scenario.distance_2d
        d_2d_in = self.scenario.distance_2d_in
        d_2d_out = self.scenario.distance_2d_out
        # Checking total 3D distances
        ut_loc = self.scenario.ut_loc
        bs_loc = self.scenario.bs_loc
        bs_loc = tf.expand_dims(bs_loc, axis=2)
        ut_loc = tf.expand_dims(ut_loc, axis=1)
        d_3d_ref = tf.sqrt(tf.reduce_sum(tf.square(ut_loc-bs_loc), axis=3))
        max_err = tf.reduce_max(tf.abs(d_3d - d_3d_ref)/d_3d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking 3D indoor + outdoor = total
        max_err = tf.reduce_max(tf.abs(d_3d-d_3d_in-d_3d_out)/d_3d)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking total 2D distances
        ut_loc = self.scenario.ut_loc
        bs_loc = self.scenario.bs_loc
        bs_loc = tf.expand_dims(bs_loc, axis=2)
        ut_loc = tf.expand_dims(ut_loc, axis=1)
        d_2d_ref = tf.sqrt(tf.reduce_sum(tf.square(ut_loc[:,:,:,:2]-bs_loc[:,:,:,:2]), axis=3))
        max_err = tf.reduce_max(tf.abs(d_2d - d_2d_ref)/d_2d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking 2D indoor + outdoor = total
        max_err = tf.reduce_max(tf.abs(d_2d-d_2d_in-d_2d_out)/d_2d)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking indoor/outdoor 2d/3d basic proportionality
        ratio_2d = d_2d_in/d_2d
        ratio_3d = d_3d_in/d_3d
        max_err = tf.reduce_max(tf.abs(ratio_2d-ratio_3d))
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)

    def test_get_param(self):
        """Test the get_param() function"""
        # Test if muDSc is correctly extracted from the file (RMa)
        param_tensor_ref = np.zeros([TestScenario.BATCH_SIZE,
                                        TestScenario.NB_BS, TestScenario.NB_UT])
        indoor = np.tile(np.expand_dims(self.scenario.indoor.numpy(), axis=1),
                            [1, TestScenario.NB_BS, 1])
        indoor_index = np.where(indoor)
        los_index = np.where(self.scenario.los.numpy())
        nlos_index = np.where(np.logical_not(self.scenario.los.numpy()))
        param_tensor_ref[los_index] = -7.49
        param_tensor_ref[nlos_index] = -7.43
        param_tensor_ref[indoor_index] = -7.47
        #
        param_tensor = self.scenario.get_param('muDSc').numpy()
        max_err = np.max(np.abs(param_tensor-param_tensor_ref))
        self.assertLessEqual(max_err, 1e-6)
    
    def test_calibration_params(self):
        """Test the calibration parameters get method and
        corresponding classes"""
        # Test if the calibration parameters are correctly acquired
        # from the scenario class. The values are those of the
        # 3GPP TR 38.901 V18.0.

        rdtype = tf.float32
        parameters_nearfield_ref = channel.tr38901.ScenarioCalibrationParameters(
                        min_bs_ut_dist = tf.constant(35., rdtype),
                        isd = tf.constant(1732., rdtype),
                        bs_height = tf.constant(35., rdtype),
                        min_ut_height = tf.constant(1.5, rdtype),
                        max_ut_height = tf.constant(1.5, rdtype),
                        indoor_probability = tf.constant(0.8, rdtype),
                        min_ut_velocity = tf.constant(3./3.6, rdtype),
                        max_ut_velocity = tf.constant(3./3.6, rdtype),
                    )
        
        parameters_farfield_ref = channel.tr38901.ScenarioCalibrationParameters(
                        min_bs_ut_dist = tf.constant(35., rdtype),
                        isd = tf.constant(1732., rdtype),
                        bs_height = tf.constant(35., rdtype),
                        min_ut_height = tf.constant(1.5, rdtype),
                        max_ut_height = tf.constant(1.5, rdtype),
                        indoor_probability = tf.constant(0.8, rdtype),
                        min_ut_velocity = tf.constant(3./3.6, rdtype),
                        max_ut_velocity = tf.constant(3./3.6, rdtype),
                    )
        
        for nearfield in [True, False]:
            parameters = self.scenario.get_calibration_parameters(nearfield=nearfield)
            if nearfield:
                parameters_ref = parameters_nearfield_ref
            else:
                parameters_ref = parameters_farfield_ref
            
            self.assertIsInstance(parameters, channel.tr38901.ScenarioCalibrationParameters)
            self.assertEqual(parameters.min_bs_ut_dist.numpy(),
                             parameters_ref.min_bs_ut_dist.numpy())
            self.assertEqual(parameters.isd.numpy(),
                             parameters_ref.isd.numpy())
            self.assertEqual(parameters.bs_height.numpy(),
                             parameters_ref.bs_height.numpy())
            self.assertEqual(parameters.min_ut_height.numpy(),
                             parameters_ref.min_ut_height.numpy())
            self.assertEqual(parameters.max_ut_height.numpy(),
                             parameters_ref.max_ut_height.numpy())
            self.assertEqual(parameters.indoor_probability.numpy(),
                             parameters_ref.indoor_probability.numpy())
            self.assertEqual(parameters.min_ut_velocity.numpy(),
                             parameters_ref.min_ut_velocity.numpy())
            self.assertEqual(parameters.max_ut_velocity.numpy(),
                             parameters_ref.max_ut_velocity.numpy())
            self.assertEqual(parameters.min_ut_velocity_indoor,
                             parameters_ref.min_ut_velocity_indoor)
            self.assertEqual(parameters.max_ut_velocity_indoor,
                             parameters_ref.max_ut_velocity_indoor)
            self.assertEqual(parameters.residential_probability,
                             parameters_ref.residential_probability)
            
    def test_release_numbers(self):
        """Test different release numbers"""
        release_numbers = ["18", "19"]

        scenarios_dict = {
            "uma": channel.tr38901.UMaScenario,
            "umi": channel.tr38901.UMiScenario,
            "inho": channel.tr38901.InHOpenOfficeScenario,
        }
        for release_number in release_numbers:
            for scenario_name, scenario_class in scenarios_dict.items():
                batch_size = TestScenario.BATCH_SIZE
                nb_bs = TestScenario.NB_BS
                nb_ut = TestScenario.NB_UT
                fc = TestScenario.CARRIER_FREQUENCY
                h_ut = TestScenario.H_UT
                h_bs = TestScenario.H_BS

                ut_orientations = tf.zeros([batch_size, nb_ut])
                bs_orientations = tf.zeros([batch_size, nb_ut])
                ut_velocities = tf.zeros([batch_size, nb_ut])

                params_dics = {"carrier_frequency": fc,
                               "ut_array": self.ut_array,
                               "bs_array": self.bs_array,
                               "direction": "uplink",
                               "release_number": release_number}
                if scenario_name not in ["rma"]:
                    params_dics["o2i_model"] = "low"
                self.scenario = scenario_class(**params_dics)
                
                ut_loc = generate_random_loc(batch_size, nb_ut, (100,2000),
                                            (100,2000), (h_ut, h_ut))
                bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                                    (0,100), (h_bs, h_bs))

                in_state = generate_random_bool(batch_size, nb_ut, 0.5)
                self.scenario.set_topology(ut_loc, bs_loc, ut_orientations,
                                        bs_orientations, ut_velocities, in_state)

                self.assertEqual(self.scenario.release_number, release_number)

    def test_calibration_mode(self):
        """Test different calibration modes"""
        calibration_modes = [True, False]

        scenarios_dict = {
            "rma": channel.tr38901.RMaScenario,
            "uma": channel.tr38901.UMaScenario,
            "umi": channel.tr38901.UMiScenario,
            "inho": channel.tr38901.InHOpenOfficeScenario,
        }
        for calibration_mode in calibration_modes:
            for scenario_name, scenario_class in scenarios_dict.items():
                batch_size = TestScenario.BATCH_SIZE
                nb_bs = TestScenario.NB_BS
                nb_ut = TestScenario.NB_UT
                fc = TestScenario.CARRIER_FREQUENCY
                h_ut = TestScenario.H_UT
                h_bs = TestScenario.H_BS

                ut_orientations = tf.zeros([batch_size, nb_ut])
                bs_orientations = tf.zeros([batch_size, nb_ut])
                ut_velocities = tf.zeros([batch_size, nb_ut])

                params_dics = {"carrier_frequency": fc,
                               "ut_array": self.ut_array,
                               "bs_array": self.bs_array,
                               "direction": "uplink",
                               "calibration_mode": calibration_mode}
                if scenario_name not in ["rma"]:
                    params_dics["o2i_model"] = "low"
                self.scenario = scenario_class(**params_dics)
                
                ut_loc = generate_random_loc(batch_size, nb_ut, (100,2000),
                                            (100,2000), (h_ut, h_ut))
                bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                                    (0,100), (h_bs, h_bs))

                in_state = generate_random_bool(batch_size, nb_ut, 0.5)
                self.scenario.set_topology(ut_loc, bs_loc, ut_orientations,
                                        bs_orientations, ut_velocities, in_state)

                self.assertEqual(self.scenario.calibration_mode, calibration_mode)
