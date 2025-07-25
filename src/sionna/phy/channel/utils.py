# pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
# pylint: disable=line-too-long, too-many-lines
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Utility functions for the channel module"""

import warnings
import tensorflow as tf

from sionna.phy.utils import expand_to_rank, sample_bernoulli
from sionna.phy import PI, config, dtypes


def subcarrier_frequencies(num_subcarriers,
                           subcarrier_spacing,
                           precision=None):
    # pylint: disable=line-too-long
    r"""
    Compute the baseband frequencies of ``num_subcarrier`` subcarriers spaced by
    ``subcarrier_spacing``, i.e.,

    >>> # If num_subcarrier is even:
    >>> frequencies = [-num_subcarrier/2, ..., 0, ..., num_subcarrier/2-1] * subcarrier_spacing
    >>>
    >>> # If num_subcarrier is odd:
    >>> frequencies = [-(num_subcarrier-1)/2, ..., 0, ..., (num_subcarrier-1)/2] * subcarrier_spacing


    Input
    ------
    num_subcarriers : `int`
        Number of subcarriers

    subcarrier_spacing : `float`
        Subcarrier spacing [Hz]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
        frequencies : [``num_subcarrier``], `tf.float`
            Baseband frequencies of subcarriers
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    start = - (num_subcarriers // 2)
    if num_subcarriers % 2 == 0:
        limit = num_subcarriers // 2
    else:
        limit = num_subcarriers // 2 + 1

    frequencies = tf.range(start=start,
                           limit=limit,
                           dtype=rdtype)
    frequencies = frequencies*subcarrier_spacing
    return frequencies


def time_frequency_vector(num_samples,
                          sample_duration,
                          precision=None):
    # pylint: disable=line-too-long
    r"""
    Compute the time and frequency vector for a given number of samples
    and duration per sample in normalized time unit.

    >>> t = tf.cast(tf.linspace(-n_min, n_max, num_samples), dtype) * sample_duration
    >>> f = tf.cast(tf.linspace(-n_min, n_max, num_samples), dtype) * 1/(sample_duration*num_samples)

    Input
    ------
    num_samples : `int`
        Number of samples

    sample_duration : `float`
        Sample duration in normalized time

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    t : [``num_samples``], `tf.float`
        Time vector

    f : [``num_samples``], `tf.float`
        Frequency vector
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
    num_samples = int(num_samples)

    if tf.math.mod(num_samples, 2) == 0:  # if even
        n_min = tf.cast(-(num_samples) / 2, dtype=tf.int32)
        n_max = tf.cast((num_samples) / 2 - 1, dtype=tf.int32)
    else:  # if odd
        n_min = tf.cast(-(num_samples-1) / 2, dtype=tf.int32)
        n_max = tf.cast((num_samples+1) / 2 - 1, dtype=tf.int32)

    # Time vector
    t = tf.cast(tf.linspace(n_min, n_max, num_samples), rdtype) \
        * tf.cast(sample_duration, rdtype)

    # Frequency vector
    df = tf.cast(1.0/sample_duration, rdtype)/tf.cast(num_samples, rdtype)
    f = tf.cast(tf.linspace(n_min, n_max, num_samples), rdtype) \
        * tf.cast(df, rdtype)

    return t, f


def time_lag_discrete_time_channel(bandwidth,
                                   maximum_delay_spread=3e-6):
    # pylint: disable=line-too-long
    r"""
    Compute the smallest and largest time-lag for the descrete complex baseband
    channel, i.e., :math:`L_{\text{min}}` and :math:`L_{\text{max}}`.

    The smallest time-lag (:math:`L_{\text{min}}`) returned is always -6, as this value
    was found small enough for all models included in Sionna.

    The largest time-lag (:math:`L_{\text{max}}`) is computed from the ``bandwidth``
    and ``maximum_delay_spread`` as follows:

    .. math::
        L_{\text{max}} = \lceil W \tau_{\text{max}} \rceil + 6

    where :math:`L_{\text{max}}` is the largest time-lag, :math:`W` the ``bandwidth``,
    and :math:`\tau_{\text{max}}` the ``maximum_delay_spread``.

    The default value for the ``maximum_delay_spread`` is 3us, which was found
    to be large enough to include most significant paths with all channel models
    included in Sionna assuming a nominal delay spread of 100ns.

    Note
    ----
    The values of :math:`L_{\text{min}}` and :math:`L_{\text{max}}` computed
    by this function are only recommended values.
    :math:`L_{\text{min}}` and :math:`L_{\text{max}}` should be set according to
    the considered channel model. For OFDM systems, one also needs to be careful
    that the effective length of the complex baseband channel is not larger than
    the cyclic prefix length.

    Input
    ------
    bandwidth : `float`
        Bandwith (:math:`W`) [Hz]

    maximum_delay_spread : `float`, (default 3e-6)
        Maximum delay spread [s]

    Output
    -------
    l_min : `int`
        Smallest time-lag (:math:`L_{\text{min}}`) for the descrete complex baseband
        channel. Set to -6, , as this value was found small enough for all models
        included in Sionna.

    l_max : `int`
        Largest time-lag (:math:`L_{\text{max}}`) for the descrete complex baseband
        channel
    """
    l_min = tf.cast(-6, tf.int32)
    l_max = tf.math.ceil(maximum_delay_spread*bandwidth) + 6
    l_max = tf.cast(l_max, tf.int32)
    return l_min, l_max


def cir_to_ofdm_channel(frequencies,
                        a,
                        tau,
                        normalize=False):
    # pylint: disable=line-too-long
    r"""
    Compute the frequency response of the channel at ``frequencies``

    Given a channel impulse response
    :math:`(a_{m}, \tau_{m}), 0 \leq m \leq M-1` (inputs ``a`` and ``tau``),
    the channel frequency response for the frequency :math:`f`
    is computed as follows:

    .. math::
        \widehat{h}(f) = \sum_{m=0}^{M-1} a_{m} e^{-j2\pi f \tau_{m}}

    Input
    ------
    frequencies : [fft_size], `tf.float`
        Frequencies at which to compute the channel response

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], `tf.complex`
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], `tf.float`
        Path delays

    normalize : `bool`, (default `False`)
        If set to `True`, the channel is normalized over the resource grid

    Output
    -------
    h_f : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], `tf.complex`
        Channel frequency responses at ``frequencies``
    """

    real_dtype = tau.dtype

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4)
        # Broadcast is not supported yet by TF for such high rank tensors.
        # We therefore do part of it manually
        tau = tf.tile(tau, [1, 1, 1, 1, a.shape[4], 1])

    # Add a time samples dimension for broadcasting
    tau = tf.expand_dims(tau, axis=6)

    # Bring all tensors to broadcastable shapes
    tau = tf.expand_dims(tau, axis=-1)
    h = tf.expand_dims(a, axis=-1)
    frequencies = expand_to_rank(frequencies, tf.rank(tau), axis=0)

    # Compute the Fourier transforms of all cluster taps
    # Exponential component
    e = tf.exp(tf.complex(tf.constant(0, real_dtype),
                          -2*PI*frequencies*tau))
    h_f = h*e
    # Sum over all clusters to get the channel frequency responses
    h_f = tf.reduce_sum(h_f, axis=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per resource grid is one.
        # Average over TX antennas, RX antennas, OFDM symbols and
        # subcarriers.
        c = tf.reduce_mean(tf.square(tf.abs(h_f)), axis=(2, 4, 5, 6),
                           keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.constant(0., real_dtype))
        h_f = tf.math.divide_no_nan(h_f, c)

    return h_f


def cir_to_time_channel(bandwidth,
                        a,
                        tau,
                        l_min,
                        l_max,
                        normalize=False):
    # pylint: disable=line-too-long
    r"""
    Compute the channel taps forming the discrete complex-baseband
    representation of the channel from the channel impulse response
    (``a``, ``tau``)

    This function assumes that a sinc filter is used for pulse shaping and receive
    filtering. Therefore, given a channel impulse response
    :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1`, the channel taps
    are computed as follows:

    .. math::
        \bar{h}_{b, \ell}
        = \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
            \text{sinc}\left( \ell - W\tau_{m} \right)

    for :math:`\ell` ranging from ``l_min`` to ``l_max``, and where :math:`W` is
    the ``bandwidth``.

    Input
    ------
    bandwidth : `float`
        Bandwidth [Hz]

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], `tf.complex`
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], `tf.float`
        Path delays [s]

    l_min : `int`
        Smallest time-lag for the discrete complex baseband channel (:math:`L_{\text{min}}`)

    l_max : `int`
        Largest time-lag for the discrete complex baseband channel (:math:`L_{\text{max}}`)

    normalize : `bool`, (default `False`)
        If set to `True`, the channel is normalized over the block size
        to ensure unit average energy per time step.

    Output
    -------
    hm :  [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], `tf.complex`
        Channel taps coefficients
    """

    real_dtype = tau.dtype

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4)
        # Broadcast is not supported by TF for such high rank tensors.
        # We therefore do part of it manually
        tau = tf.tile(tau, [1, 1, 1, 1, a.shape[4], 1])

    # Add a time samples dimension for broadcasting
    tau = tf.expand_dims(tau, axis=6)

    # Time lags for which to compute the channel taps
    l = tf.range(l_min, l_max+1, dtype=real_dtype)

    # Bring tau and l to broadcastable shapes
    tau = tf.expand_dims(tau, axis=-1)
    l = expand_to_rank(l, tau.shape.rank, axis=0)

    # sinc pulse shaping
    g = tf.experimental.numpy.sinc(l-tau*bandwidth)
    g = tf.complex(g, tf.constant(0., real_dtype))
    a = tf.expand_dims(a, axis=-1)

    # For every tap, sum the sinc-weighted coefficients
    hm = tf.reduce_sum(a*g, axis=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per block is one.
        # The total energy of a channel response is the sum of the squared
        # norm over the channel taps.
        # Average over block size, RX antennas, and TX antennas
        c = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(hm)),
                                         axis=6, keepdims=True),
                           axis=(2, 4, 5), keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.constant(0., real_dtype))
        hm = tf.math.divide_no_nan(hm, c)

    return hm


def time_to_ofdm_channel(h_t,
                         rg,
                         l_min):
    # pylint: disable=line-too-long
    r"""
    Compute the channel frequency response from the discrete complex-baseband
    channel impulse response

    Given a discrete complex-baseband channel impulse response
    :math:`\bar{h}_{b,\ell}`, for :math:`\ell` ranging from :math:`L_\text{min}\le 0`
    to :math:`L_\text{max}`, the discrete channel frequency response is computed as

    .. math::

        \hat{h}_{b,n} = \sum_{k=0}^{L_\text{max}} \bar{h}_{b,k} e^{-j \frac{2\pi kn}{N}} + \sum_{k=L_\text{min}}^{-1} \bar{h}_{b,k} e^{-j \frac{2\pi n(N+k)}{N}}, \quad n=0,\dots,N-1

    where :math:`N` is the FFT size and :math:`b` is the time step.

    This function only produces one channel frequency response per OFDM symbol, i.e.,
    only values of :math:`b` corresponding to the start of an OFDM symbol (after
    cyclic prefix removal) are considered.

    Input
    ------
    h_t : [...num_time_steps,l_max-l_min+1], `tf.complex`
        Tensor of discrete complex-baseband channel impulse responses

    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        Resource grid

    l_min : `int`
        Smallest time-lag for the discrete complex baseband
        channel impulse response (:math:`L_{\text{min}}`)

    Output
    ------
    h_f : [...,num_ofdm_symbols,fft_size], `tf.complex`
        Tensor of discrete complex-baseband channel frequency responses

    Note
    ----
    Note that the result of this function is generally different from the
    output of :meth:`~sionna.phy.channel.utils.cir_to_ofdm_channel` because
    the discrete complex-baseband channel impulse response is truncated
    (see :meth:`~sionna.phy.channel.utils.cir_to_time_channel`). This effect
    can be observed in the example below.

    Examples
    --------
    .. code-block:: Python

        # Setup resource grid and channel model
        sm = StreamManagement(np.array([[1]]), 1)
        rg = ResourceGrid(num_ofdm_symbols=1,
                          fft_size=1024,
                          subcarrier_spacing=15e3)
        tdl = TDL("A", 100e-9, 3.5e9)

        # Generate CIR
        cir = tdl(batch_size=1, num_time_steps=1, sampling_frequency=rg.bandwidth)

        # Generate OFDM channel from CIR
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = tf.squeeze(cir_to_ofdm_channel(frequencies, *cir, normalize=True))

        # Generate time channel from CIR
        l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
        h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min=l_min, l_max=l_max, normalize=True)

        # Generate OFDM channel from time channel
        h_freq_hat = tf.squeeze(time_to_ofdm_channel(h_time, rg, l_min))

        # Visualize results
        plt.figure()
        plt.plot(np.real(h_freq), "-")
        plt.plot(np.real(h_freq_hat), "--")
        plt.plot(np.imag(h_freq), "-")
        plt.plot(np.imag(h_freq_hat), "--")
        plt.xlabel("Subcarrier index")
        plt.ylabel(r"Channel frequency response")
        plt.legend(["OFDM Channel (real)", "OFDM Channel from time (real)", "OFDM Channel (imag)", "OFDM Channel from time (imag)"])

    .. image:: ../figures/time_to_ofdm_channel.png
    """
    # Totla length of an OFDM symbol including cyclic prefix
    ofdm_length = rg.fft_size + rg.cyclic_prefix_length

    # Downsample the impulse respons to one sample per OFDM symbol
    h_t = h_t[..., rg.cyclic_prefix_length:rg.num_time_samples:ofdm_length, :]

    # Pad channel impulse response with zeros to the FFT size
    pad_dims = rg.fft_size - tf.shape(h_t)[-1]
    pad_shape = tf.concat([tf.shape(h_t)[:-1], [pad_dims]], axis=-1)
    h_t = tf.concat([h_t, tf.zeros(pad_shape, dtype=h_t.dtype)], axis=-1)

    # Circular shift of negative time lags so that the channel impulse reponse
    # starts with h_{b,0}
    h_t = tf.roll(h_t, l_min, axis=-1)

    # Compute FFT
    h_f = tf.signal.fft(h_t)

    # Move the zero subcarrier to the center of the spectrum
    h_f = tf.signal.fftshift(h_f, axes=-1)

    return h_f


def deg_2_rad(x):
    r"""
    Convert degree to radian

    Input
    ------
    x : Tensor, `tf.float`
        Angles in degree

    Output
    -------
    y : Tensor, `tf.float`
        Angles ``x`` converted to radian
    """
    return x*tf.constant(PI/180.0, x.dtype)


def rad_2_deg(x):
    r"""
    Convert radian to degree

    Input
    ------
    x : Tensor, `tf.float`
        Angles in radian

    Output
    -------
    y : Tensor, `tf.float`
        Angles ``x`` converted to degree
    """
    return x*tf.constant(180.0/PI, x.dtype)


def wrap_angle_0_360(angle):
    r"""
    Wrap ``angle`` to (0,360)

    Input
    ------
    angle : Tensor, `tf.float`
        Input to wrap

    Output
    -------
    y : Tensor, `tf.float`
        ``angle`` wrapped to (0,360)
    """
    return tf.math.mod(angle, 360.)


def drop_uts_in_sector(batch_size,
                       num_ut,
                       min_bs_ut_dist,
                       isd,
                       bs_height=0.,
                       ut_height=0.,
                       precision=None):
    r"""
    Sample UT locations uniformly at random within a sector

    The sector from which UTs are sampled is shown in the following figure.
    The BS is assumed to be located at the origin (0,0) of the coordinate
    system.

    .. figure:: ../figures/drop_uts_in_sector.png
        :align: center
        :scale: 30%

    Input
    --------
    batch_size : `int`
        Batch size

    num_ut : `int`
        Number of UTs to sample per batch example

    min_bs_ut_dist : `tf.float`
        Minimum BS-UT distance [m]

    isd : `tf.float`
        Inter-site distance, i.e., the distance between two adjacent BSs [m]

    bs_height : `tf.float`, (default 0)
        BS height, i.e., distance between the BS and the X-Y plane [m]

    ut_height : `tf.float`
        UT height, i.e., distance between the UT and the X-Y plane [m]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 2], `tf.float`
        UT locations in the X-Y plane
    """

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Cast to rdtype
    d_min = tf.cast(min_bs_ut_dist, rdtype)
    bs_height = tf.cast(bs_height, rdtype)
    ut_height = tf.cast(ut_height, rdtype)

    # Force the minimum BS-UT distance >= their height difference
    # d_min = tf.maximum(d_min, tf.abs(bs_height - ut_height))

    r = tf.cast(isd*0.5, rdtype)

    # Minimum squared distance between BS and UT on the X-Y plane
    # r_min2 = d_min**2 - (bs_height - ut_height)**2
    r_min2 = d_min**2

    # Angles from (-pi/6, pi/6), covering half of the sector and denoted by
    # alpha_half, are randomly sampled for all UTs.
    # Then, the maximum distance of UTs from the BS, denoted by r_max,
    # is computed for each angle.
    # For each angles, UT positions are sampled such that their *squared* from
    # the BTS is uniformly sampled from the range (r_min**2, r_max**2)
    # Each UT is then randomly and uniformly pushed to a half of the sector
    # by adding either PI/6 or PI/2 to the angle alpha_half

    # Sample angles for half of the sector (which half will be decided randomly)
    alpha_half = config.tf_rng.uniform(shape=[batch_size, num_ut],
                                       minval=-PI/6.,
                                       maxval=PI/6.,
                                       dtype=rdtype)

    # Maximum distance (computed on the X-Y plane) from BS to a point in
    # the sector, at each angle in alpha_half
    r_max = r / tf.math.cos(alpha_half)

    # To ensure the UT distribution to be uniformly distributed across the
    # sector, we sample positions such that their *squared* distance from
    # the BS is uniformly distributed within (r_min**2, r_max**2)
    distance2 = config.tf_rng.uniform(shape=[batch_size, num_ut],
                                      minval=r_min2,
                                      maxval=r_max**2,
                                      dtype=rdtype)
    distance = tf.sqrt(distance2)

    # Randomly assign the UTs to one of the two halves of the sector
    side = sample_bernoulli([batch_size, num_ut],
                            tf.cast(0.5, rdtype),
                            precision=precision)
    side = tf.cast(side, rdtype)
    side = 2.*side + 1.
    alpha = alpha_half + side*PI/6.

    # Compute UT locations in the X-Y plane
    ut_loc = tf.stack([distance*tf.math.cos(alpha),
                       distance*tf.math.sin(alpha)], axis=-1)

    return ut_loc


def set_3gpp_scenario_parameters(scenario,
                                 mode=None,
                                 min_bs_ut_dist=None,
                                 isd=None,
                                 bs_height=None,
                                 min_ut_height=None,
                                 max_ut_height=None,
                                 indoor_probability=None,
                                 min_ut_velocity=None,
                                 max_ut_velocity=None,
                                 precision=None):
    r"""
    Set valid parameters for a specified 3GPP system level ``scenario``
    (RMa, UMi, or UMa)

    If a parameter is given, then it is returned. If it is set to `None`,
    then a parameter valid according to the chosen scenario is returned
    (see [TR38901]_).

    Input
    --------
    scenario : "uma" | "umi" | "rma" | inh-open-office | "uma-calibration" | "umi-calibration | inh-open-office-calibration"
        System level model scenario

    mode : "phase1" | "phase2-config1" | "phase2-config2" | "nearfield-config1A" | "nearfield-config1B" | "nearfield-config2A" | "nearfield-config2B"
        Mode of calibration.

    min_bs_ut_dist : `None` (default) | `tf.float`
        Minimum BS-UT distance [m]

    isd : `None` (default) | `tf.float`
        Inter-site distance [m]

    bs_height : `None` (default) | `tf.float`
        BS elevation [m]

    min_ut_height : `None` (default) | `tf.float`
        Minimum UT elevation [m]

    max_ut_height : `None` (default) | `tf.float`
        Maximum UT elevation [m]

    indoor_probability : `None` (default) | `tf.float`
        Probability of a UT to be indoor

    min_ut_velocity : `None` (default) | `tf.float`
        Minimum UT velocity [m/s]

    max_ut_velocity : `None` (default) | `tf.float`
        Maximim UT velocity [m/s]

    precision : str, `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`,
        :py:attr:`~sionna.phy.config.precision` is used.

    Output
    --------
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
        Minimum UT velocity [m/s]

    max_ut_velocity : `tf.float`
        Maximim UT velocity [m/s]
    """

    assert scenario in ('umi', 'uma', 'rma', 'inh-open-office', 'umi-calibration', 'uma-calibration', 'inh-open-office-calibration'), \
        "`scenario` must be one of 'umi', 'uma', 'rma', 'inh-open-office', 'umi-calibration', 'uma-calibration', 'inh-open-office-calibration'"

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Default values for scenario parameters.
    # All distances and heights are in meters
    # All velocities are in meters per second.
    if mode is not None and mode.startswith("nearfield"):
        default_scenario_par = {
        'umi-calibration': {
                'min_bs_ut_dist': tf.constant(10., rdtype),
                'isd': tf.constant(200., rdtype),
                'bs_height': tf.constant(10., rdtype),
                'min_ut_height': tf.constant(1.5, rdtype),
                'max_ut_height': tf.constant(1.5, rdtype),
                'indoor_probability': tf.constant(0.0, rdtype),
                'min_ut_velocity': tf.constant(3./3.6, rdtype),
                'max_ut_velocity': tf.constant(3./3.6, rdtype)
            },
        'inh-open-office-calibration': {
                'min_bs_ut_dist': tf.constant(0., rdtype),
                'isd': tf.constant(20., rdtype),
                'bs_height': tf.constant(3., rdtype),
                'min_ut_height': tf.constant(1., rdtype),
                'max_ut_height': tf.constant(1., rdtype),
                # indoor probability is 0 to prevent the o2i from being applied
                'indoor_probability': tf.constant(0.0, rdtype),
                'min_ut_velocity': tf.constant(3./3.6, rdtype),
                'max_ut_velocity': tf.constant(3./3.6, rdtype)
            }
        }
    else:
        default_scenario_par = {
            'umi': {
                'min_bs_ut_dist': tf.constant(10., rdtype),
                'isd': tf.constant(200., rdtype),
                'bs_height': tf.constant(10., rdtype),
                'min_ut_height': tf.constant(1.5, rdtype),
                'max_ut_height': tf.constant(1.5, rdtype),
                'indoor_probability': tf.constant(0.8, rdtype),
                'min_ut_velocity': tf.constant(0.0, rdtype),
                'max_ut_velocity': tf.constant(0.0, rdtype)
            },
            'umi-calibration': {
                'min_bs_ut_dist': tf.constant(10., rdtype),
                'isd': tf.constant(200., rdtype),
                'bs_height': tf.constant(10., rdtype),
                'min_ut_height': tf.constant(1.5, rdtype),
                'max_ut_height': tf.constant(1.5, rdtype),
                'indoor_probability': tf.constant(0.8, rdtype),
                'min_ut_velocity': tf.constant(3./3.6, rdtype),
                'max_ut_velocity': tf.constant(3./3.6, rdtype)
            },
            'uma': {
                'min_bs_ut_dist': tf.constant(35., rdtype),
                'isd': tf.constant(500., rdtype),
                'bs_height': tf.constant(25., rdtype),
                'min_ut_height': tf.constant(1.5, rdtype),
                'max_ut_height': tf.constant(1.5, rdtype),
                'indoor_probability': tf.constant(0.8, rdtype),
                'min_ut_velocity': tf.constant(0.0, rdtype),
                'max_ut_velocity': tf.constant(0.0, rdtype),
            },
            'uma-calibration': {
                'min_bs_ut_dist': tf.constant(35., rdtype),
                'isd': tf.constant(500., rdtype),
                'bs_height': tf.constant(25., rdtype),
                'min_ut_height': tf.constant(1.5, rdtype),
                'max_ut_height': tf.constant(1.5, rdtype),
                'indoor_probability': tf.constant(0.8, rdtype),
                'min_ut_velocity': tf.constant(3./3.6, rdtype),
                'max_ut_velocity': tf.constant(3./3.6, rdtype),
            },
            'rma': {
                'min_bs_ut_dist': tf.constant(35., rdtype),
                'isd': tf.constant(5000., rdtype),
                'bs_height': tf.constant(35., rdtype),
                'min_ut_height': tf.constant(1.5, rdtype),
                'max_ut_height': tf.constant(1.5, rdtype),
                'indoor_probability': tf.constant(0.5, rdtype),
                'min_ut_velocity': tf.constant(0.0, rdtype),
                'max_ut_velocity': tf.constant(0.0, rdtype)
            },
            'inh-open-office': {
                'min_bs_ut_dist': tf.constant(0., rdtype),
                'isd': tf.constant(20., rdtype),
                'bs_height': tf.constant(3., rdtype),
                'min_ut_height': tf.constant(1., rdtype),
                'max_ut_height': tf.constant(1., rdtype),
                # indoor probability is 0 to prevent the o2i from being applied
                'indoor_probability': tf.constant(0.0, rdtype),
                'min_ut_velocity': tf.constant(3./3.6, rdtype),
                'max_ut_velocity': tf.constant(3./3.6, rdtype)
            },
            'inh-open-office-calibration': {
                'min_bs_ut_dist': tf.constant(0., rdtype),
                'isd': tf.constant(20., rdtype),
                'bs_height': tf.constant(3., rdtype),
                'min_ut_height': tf.constant(1., rdtype),
                'max_ut_height': tf.constant(1., rdtype),
                # indoor probability is 0 to prevent the o2i from being applied
                'indoor_probability': tf.constant(0.0, rdtype),
                'min_ut_velocity': tf.constant(3./3.6, rdtype),
                'max_ut_velocity': tf.constant(3./3.6, rdtype)
            }
        }
    # Setting the scenario parameters
    if min_bs_ut_dist is None:
        min_bs_ut_dist = default_scenario_par[scenario]['min_bs_ut_dist']
    if isd is None:
        isd = default_scenario_par[scenario]['isd']
    if bs_height is None:
        bs_height = default_scenario_par[scenario]['bs_height']
    if min_ut_height is None:
        min_ut_height = default_scenario_par[scenario]['min_ut_height']
    if max_ut_height is None:
        max_ut_height = default_scenario_par[scenario]['max_ut_height']
    if indoor_probability is None:
        indoor_probability = default_scenario_par[scenario]['indoor_probability']
    if min_ut_velocity is None:
        min_ut_velocity = default_scenario_par[scenario]['min_ut_velocity']
    if max_ut_velocity is None:
        max_ut_velocity = default_scenario_par[scenario]['max_ut_velocity']

    return min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, \
        indoor_probability, min_ut_velocity, max_ut_velocity



def relocate_uts(ut_loc,
                 sector_id,
                 cell_loc):
    # pylint: disable=line-too-long
    r"""
    Relocate the UTs by rotating them into the sector with index ``sector_id``
    and transposing them to the cell centered on ``cell_loc``

    ``sector_id`` gives the index of the sector to which the UTs are
    rotated to. The picture below shows how the three sectors of a cell are
    indexed.

    .. figure:: ../figures/panel_array_sector_id.png
        :align: center
        :scale: 30%

        Indexing of sectors

    If ``sector_id`` is a scalar, then all UTs are relocated to the same
    sector indexed by ``sector_id``.
    If ``sector_id`` is a tensor, it should be broadcastable with
    [``batch_size``, ``num_ut``], and give the sector in which each UT or
    batch example is relocated to.

    When calling the function, ``ut_loc`` gives the locations of the UTs to
    relocate, which are all assumed to be in sector with index 0, and in the
    cell centered on the origin (0,0).

    Input
    --------
    ut_loc : [batch_size, num_ut, 2], `tf.float`
        UTs locations in the X-Y plan

    sector_id : Tensor broadcastable with [batch_size, num_ut], `int`
        Indexes of the sector to which to relocate the UTs

    cell_loc : Tensor broadcastable with [batch_size, num_ut], `tf.float`
        Center of the cell to which to transpose the UTs

    Output
    ------
    ut_loc : [batch_size, num_ut, 2], `tf.float`
        Relocated UTs locations in the X-Y plan
    """

    # Expand the rank of sector_id such that is is broadcastable with
    # (batch size, num_ut)
    sector_id = tf.cast(sector_id, ut_loc.dtype)
    sector_id = expand_to_rank(sector_id, 2, 0)

    # Expant
    cell_loc = tf.cast(cell_loc, ut_loc.dtype)
    cell_loc = expand_to_rank(cell_loc, tf.rank(ut_loc), 0)

    # Rotation matrix tensor, broadcastable with [batch size, num uts, 2, 2]
    rotation_angle = sector_id*2.*PI/3.0
    rotation_matrix = tf.stack([tf.math.cos(rotation_angle),
                                -tf.math.sin(rotation_angle),
                                tf.math.sin(rotation_angle),
                                tf.math.cos(rotation_angle)],
                               axis=-1)
    rotation_matrix = tf.reshape(rotation_matrix,
                                 tf.concat([tf.shape(rotation_angle),
                                            [2, 2]], axis=-1))
    rotation_matrix = tf.cast(rotation_matrix, ut_loc.dtype)

    # Applying the rotation matrix
    ut_loc = tf.expand_dims(ut_loc, axis=-1)
    ut_loc_rotated = tf.squeeze(rotation_matrix@ut_loc, axis=-1)

    # Translate to the BS location
    ut_loc_rotated_translated = ut_loc_rotated + cell_loc

    return ut_loc_rotated_translated


def generate_uts_topology(batch_size,
                          num_ut,
                          drop_area,
                          cell_loc_xy,
                          min_bs_ut_dist,
                          isd,
                          min_ut_height,
                          max_ut_height,
                          indoor_probability,
                          min_ut_velocity,
                          max_ut_velocity,
                          precision=None):
    # pylint: disable=line-too-long
    r"""
    Sample UTs location from a sector or a cell

    Input
    --------
    batch_size : `int`
        Batch size

    num_ut : `int`
        Number of UTs to sample per batch example

    drop_area : "sector" | "cell"
        If set to "sector", UTs are sampled from the
        sector with index 0 in the figure below

        .. figure:: ../figures/panel_array_sector_id.png
            :align: center
            :scale: 30%

    Indexing of sectors

    cell_loc_xy : Tensor broadcastable with[batch_size, num_ut, 3], `tf.float`
        Center of the cell(s)

    min_bs_ut_dist : `None` | `tf.float`
        Minimum BS-UT distance [m]

    isd : `None` | `tf.float`
        Inter-site distance [m]

    min_ut_height : `None` | `tf.float`
        Minimum UT elevation [m]

    max_ut_height : `None` | `tf.float`
        Maximum UT elevation [m]

    indoor_probability : `None` | `tf.float`
        Probability of a UT to be indoor

    min_ut_velocity : `None` | `tf.float`
        Minimum UT velocity [m/s]

    max_ut_velocity : `None` | `tf.float`
        Maximum UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], `tf.float`
        UTs locations

    ut_orientations : [batch_size, num_ut, 3], `tf.float`
        UTs orientations [radian]

    ut_velocities : [batch_size, num_ut, 3], `tf.float`
        UTs velocities [m/s]. The norm is drawn uniformly at random
        between ``min_ut_velocity`` and ``max_ut_velocity`, while the direction
        is planar and drawn uniformly at random between 0 and :math:`2\pi`

    in_state : [batch_size, num_ut], `tf.float`
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """

    assert drop_area in ('sector', 'cell'), \
        "Drop area must be either 'sector' or 'cell'"

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Randomly generating the UT locations
    ut_loc_xy = drop_uts_in_sector(batch_size,
                                   num_ut,
                                   min_bs_ut_dist,
                                   isd,
                                   precision=precision)
    if drop_area == 'sector':
        sectors = tf.constant(0, tf.int32)
    else:  # 'cell'
        sectors = config.tf_rng.uniform(shape=[batch_size, num_ut],
                                        minval=0,
                                        maxval=3,
                                        dtype=tf.int32)
    ut_loc_xy = relocate_uts(ut_loc_xy,
                             sectors,
                             cell_loc_xy)

    ut_loc_z = config.tf_rng.uniform(shape=[batch_size, num_ut, 1],
                                     minval=min_ut_height,
                                     maxval=max_ut_height,
                                     dtype=rdtype)
    ut_loc = tf.concat([ut_loc_xy,
                        ut_loc_z], axis=-1)

    # Draw random UT orientation, velocity and indoor state
    ut_orientations, ut_velocities, in_state = \
        random_ut_properties(batch_size,
                             num_ut,
                             indoor_probability,
                             min_ut_velocity,
                             max_ut_velocity,
                             precision=precision)

    return ut_loc, ut_orientations, ut_velocities, in_state


def random_ut_properties(batch_size,
                         num_ut,
                         indoor_probability,
                         min_ut_velocity,
                         max_ut_velocity,
                         precision=None):
    r"""
    Draw random values for UT velocity, orientation and indoor state

    Input
    --------
    batch_size : `int`
        Batch size

    num_ut : `int`
        Number of UTs to sample per batch example

    indoor_probability : `None` | `tf.float`
        Probability of a UT to be indoor

    min_ut_velocity : `None` | `tf.float`
        Minimum UT velocity [m/s]

    max_ut_velocity : `None` | `tf.float`
        Maximum UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------

    ut_orientations : [batch_size, num_ut, 3], `tf.float`
        UTs orientations [radian]

    ut_velocities : [batch_size, num_ut, 3], `tf.float`
        UTs velocity vectors [m/s]. The norm is drawn uniformly at random
        between ``min_ut_velocity`` and ``max_ut_velocity`, while the direction
        is planar and drawn uniformly at random between 0 and :math:`2\pi`

    in_state : [batch_size, num_ut], `tf.float`
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Randomly generating the UT indoor/outdoor state
    in_state = sample_bernoulli([batch_size, num_ut],
                                indoor_probability,
                                precision=precision)

    # Randomly generate the UT velocities
    ut_vel_angle = config.tf_rng.uniform([batch_size, num_ut],
                                         minval=-PI,
                                         maxval=PI,
                                         dtype=rdtype)
    ut_vel_norm = config.tf_rng.uniform([batch_size, num_ut],
                                        minval=min_ut_velocity,
                                        maxval=max_ut_velocity,
                                        dtype=rdtype)
    ut_velocities = tf.stack([ut_vel_norm*tf.math.cos(ut_vel_angle),
                              ut_vel_norm*tf.math.sin(ut_vel_angle),
                              tf.zeros([batch_size, num_ut], rdtype)],
                             axis=-1)

    # Randomly generate the UT orientations
    ut_bearing = config.tf_rng.uniform([batch_size, num_ut], minval=-0.5*PI,
                                       maxval=0.5*PI, dtype=rdtype)
    ut_downtilt = config.tf_rng.uniform([batch_size, num_ut], minval=-0.5*PI,
                                        maxval=0.5*PI, dtype=rdtype)
    ut_slant = config.tf_rng.uniform([batch_size, num_ut], minval=-0.5*PI,
                                     maxval=0.5*PI, dtype=rdtype)
    ut_orientations = tf.stack([ut_bearing, ut_downtilt, ut_slant], axis=-1)

    return ut_orientations, ut_velocities, in_state


def gen_single_sector_topology(batch_size,
                               num_ut,
                               scenario,
                               min_bs_ut_dist=None,
                               isd=None,
                               bs_height=None,
                               min_ut_height=None,
                               max_ut_height=None,
                               indoor_probability=None,
                               min_ut_velocity=None,
                               max_ut_velocity=None,
                               precision=None):
    # pylint: disable=line-too-long
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin and ``num_ut`` UTs randomly and uniformly dropped in a cell sector

    The following picture shows the sector from which UTs are sampled.

    .. figure:: ../figures/drop_uts_in_sector.png
        :align: center
        :scale: 30%

    UT velocity and orientation are drawn uniformly at random, whereas the BS points
    towards the center of the sector it serves.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.phy.channel.tr38901.UMi`, :class:`~sionna.phy.channel.tr38901.UMa`,
    and :class:`~sionna.phy.channel.tr38901.RMa`.

    Example
    --------
    >>> # Create antenna arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                      num_cols_per_panel = 4,
    ...                      polarization = 'dual',
    ...                      polarization_type = 'VH',
    ...                      antenna_pattern = '38.901',
    ...                      carrier_frequency = 3.5e9)
    >>>
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Create channel model
    >>> channel_model = UMi(carrier_frequency = 3.5e9,
    ...                     o2i_model = 'low',
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Generate the topology
    >>> topology = gen_single_sector_topology(batch_size = 100,
    ...                                       num_ut = 4,
    ...                                       scenario = 'umi')
    >>> # Set the topology
    >>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_sector_topology.png

    Input
    --------
    batch_size : `int`
        Batch size

    num_ut : `int`
        Number of UTs to sample per batch example

    scenario : "uma" | "umi" | "rma" | "uma-calibration" | "umi-calibration"
        System level model scenario

    min_bs_ut_dist : `None` (default) | `tf.float`
        Minimum BS-UT distance [m]

    isd : `None` (default) | `tf.float`
        Inter-site distance [m]

    bs_height : `None` (default) | `tf.float`
        BS elevation [m]

    min_ut_height : `None` (default) | `tf.float`
        Minimum UT elevation [m]

    max_ut_height : `None` (default) | `tf.float`
        Maximum UT elevation [m]

    indoor_probability : `None` (default) | `tf.float`
        Probability of a UT to be indoor

    min_ut_velocity : `None` (default) | `tf.float`
        Minimum UT velocity [m/s]

    max_ut_velocity : `None` (default) | `tf.float`
        Maximim UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], `tf.float`
        UTs locations

    bs_loc : [batch_size, 1, 3], `tf.float`
        BS location. Set to (0,0,0) for all batch examples.

    ut_orientations : [batch_size, num_ut, 3], `tf.float`
        UTs orientations [radian]

    bs_orientations : [batch_size, 1, 3], `tf.float`
        BS orientations [radian]. Oriented towards the center of the sector.

    ut_velocities : [batch_size, num_ut, 3], `tf.float`
        UTs velocities [m/s]

    in_state : [batch_size, num_ut], `tf.float`
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """
    params = set_3gpp_scenario_parameters(scenario,
                                          min_bs_ut_dist,
                                          isd,
                                          bs_height,
                                          min_ut_height,
                                          max_ut_height,
                                          indoor_probability,
                                          min_ut_velocity,
                                          max_ut_velocity,
                                          precision=precision)
    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, \
        indoor_probability, min_ut_velocity, max_ut_velocity = params

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Setting BS to (0,0,bs_height)
    bs_loc = tf.stack([tf.zeros([batch_size, 1], rdtype),
                       tf.zeros([batch_size, 1], rdtype),
                       tf.fill([batch_size, 1], bs_height)], axis=-1)

    # Setting the BS orientation such that it is downtilted towards the center
    # of the sector
    sector_center = (min_bs_ut_dist + 0.5*isd)*0.5
    bs_downtilt = 0.5*PI - tf.math.atan(sector_center/bs_height)
    bs_yaw = tf.constant(PI/3.0, rdtype)
    bs_orientation = tf.stack([tf.fill([batch_size, 1], bs_yaw),
                               tf.fill([batch_size, 1], bs_downtilt),
                               tf.zeros([batch_size, 1], rdtype)], axis=-1)
    # Generating the UTs
    ut_topology = generate_uts_topology(batch_size,
                                        num_ut,
                                        'sector',
                                        tf.zeros([2], rdtype),
                                        min_bs_ut_dist,
                                        isd,
                                        min_ut_height,
                                        max_ut_height,
                                        indoor_probability,
                                        min_ut_velocity,
                                        max_ut_velocity,
                                        precision)
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, \
        in_state


def gen_single_sector_topology_interferers(batch_size,
                                           num_ut,
                                           num_interferer,
                                           scenario,
                                           min_bs_ut_dist=None,
                                           isd=None,
                                           bs_height=None,
                                           min_ut_height=None,
                                           max_ut_height=None,
                                           indoor_probability=None,
                                           min_ut_velocity=None,
                                           max_ut_velocity=None,
                                           precision=None):
    # pylint: disable=line-too-long
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin, ``num_ut`` UTs randomly and uniformly dropped in a cell sector, and
    ``num_interferer`` interfering UTs randomly dropped in the adjacent cells

    The following picture shows how UTs are sampled

    .. figure:: ../figures/drop_uts_in_sector_interferers.png
        :align: center
        :scale: 30%

    UT velocity and orientation are drawn uniformly at random, whereas the BS points
    towards the center of the sector it serves.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.phy.channel.tr38901.UMi`, :class:`~sionna.phy.channel.tr38901.UMa`,
    and :class:`~sionna.phy.channel.tr38901.RMa`.

    In the returned ``ut_loc``, ``ut_orientations``, ``ut_velocities``, and
    ``in_state`` tensors, the first ``num_ut`` items along the axis with index
    1 correspond to the served UTs, whereas the remaining ``num_interferer``
    items correspond to the interfering UTs.

    Example
    --------
    >>> # Create antenna arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                      num_cols_per_panel = 4,
    ...                      polarization = 'dual',
    ...                      polarization_type = 'VH',
    ...                      antenna_pattern = '38.901',
    ...                      carrier_frequency = 3.5e9)
    >>>
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Create channel model
    >>> channel_model = UMi(carrier_frequency = 3.5e9,
    ...                     o2i_model = 'low',
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Generate the topology
    >>> topology = gen_single_sector_topology_interferers(batch_size = 100,
    ...                                                   num_ut = 4,
    ...                                                   num_interferer = 4,
    ...                                                   scenario = 'umi')
    >>> # Set the topology
    >>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_sector_topology_inter.png

    Input
    --------
    batch_size : `int`
        Batch size

    num_ut : `int`
        Number of UTs to sample per batch example

    num_interferer : `int`
        Number of interfeering UTs per batch example

    scenario : "uma" | "umi" | "rma" | "uma-calibration" | "umi-calibration"
        System level model scenario

    min_bs_ut_dist : `None` (default) | `tf.float`
        Minimum BS-UT distance [m]

    isd : `None` (default) | `tf.float`
        Inter-site distance [m]

    bs_height : `None` (default) | `tf.float`
        BS elevation [m]

    min_ut_height : `None` (default) | `tf.float`
        Minimum UT elevation [m]

    max_ut_height : `None` (default) | `tf.float`
        Maximum UT elevation [m]

    indoor_probability : `None` (default) | `tf.float`
        Probability of a UT to be indoor

    min_ut_velocity : `None` (default) | `tf.float`
        Minimum UT velocity [m/s]

    max_ut_velocity : `None` (default) | `tf.float`
        Maximim UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], `tf.float`
        UTs locations. The first ``num_ut`` items along the axis with index
        1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfeering UTs.

    bs_loc : [batch_size, 1, 3], `tf.float`
        BS location. Set to (0,0,0) for all batch examples.

    ut_orientations : [batch_size, num_ut, 3], `tf.float`
        UTs orientations [radian]. The first ``num_ut`` items along the
        axis with index 1 correspond to the served UTs, whereas the
        remaining ``num_interferer`` items correspond to the interfeering
        UTs.

    bs_orientations : [batch_size, 1, 3], `tf.float`
        BS orientation [radian]. Oriented towards the center of the sector.

    ut_velocities : [batch_size, num_ut, 3], `tf.float`
        UTs velocities [m/s]. The first ``num_ut`` items along the axis
        with index 1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfeering UTs.

    in_state : [batch_size, num_ut], `tf.float`
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor. The first ``num_ut`` items along the axis with
        index 1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfering UTs.
    """

    params = set_3gpp_scenario_parameters(scenario,
                                          min_bs_ut_dist,
                                          isd,
                                          bs_height,
                                          min_ut_height,
                                          max_ut_height,
                                          indoor_probability,
                                          min_ut_velocity,
                                          max_ut_velocity,
                                          precision)
    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, \
        indoor_probability, min_ut_velocity, max_ut_velocity = params

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Setting BS to (0,0,bs_height)
    bs_loc = tf.stack([tf.zeros([batch_size, 1], rdtype),
                       tf.zeros([batch_size, 1], rdtype),
                       tf.fill([batch_size, 1], bs_height)], axis=-1)

    # Setting the BS orientation such that it is downtilted towards the center
    # of the sector
    sector_center = (min_bs_ut_dist + 0.5*isd)*0.5
    bs_downtilt = 0.5*PI - tf.math.atan(sector_center/bs_height)
    bs_yaw = tf.constant(PI/3.0, rdtype)
    bs_orientation = tf.stack([tf.fill([batch_size, 1], bs_yaw),
                               tf.fill([batch_size, 1], bs_downtilt),
                               tf.zeros([batch_size, 1], rdtype)], axis=-1)

    # Generating the UTs located in the UTs served by the BS
    ut_topology = generate_uts_topology(batch_size,
                                        num_ut,
                                        'sector',
                                        tf.zeros([2], rdtype),
                                        min_bs_ut_dist,
                                        isd,
                                        min_ut_height,
                                        max_ut_height,
                                        indoor_probability,
                                        min_ut_velocity,
                                        max_ut_velocity,
                                        precision)
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    # Generating the UTs located in the adjacent cells

    # Users are randomly dropped in one of the two adjacent cells
    inter_cell_center = tf.stack([[0.0, isd],
                                  [isd*tf.math.cos(PI/6.0),
                                   isd*tf.math.sin(PI/6.0)]],
                                 axis=0)
    cell_index = config.tf_rng.uniform(shape=[batch_size, num_interferer],
                                       minval=0, maxval=2, dtype=tf.int32)
    inter_cells = tf.gather(inter_cell_center, cell_index)

    inter_topology = generate_uts_topology(batch_size,
                                           num_interferer,
                                           'cell',
                                           inter_cells,
                                           min_bs_ut_dist,
                                           isd,
                                           min_ut_height,
                                           max_ut_height,
                                           indoor_probability,
                                           min_ut_velocity,
                                           max_ut_velocity,
                                           precision)
    inter_loc, inter_orientations, inter_velocities, inter_in_state \
        = inter_topology

    ut_loc = tf.concat([ut_loc, inter_loc], axis=1)
    ut_orientations = tf.concat([ut_orientations, inter_orientations], axis=1)
    ut_velocities = tf.concat([ut_velocities, inter_velocities], axis=1)
    in_state = tf.concat([in_state, inter_in_state], axis=1)

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, \
        in_state


def exp_corr_mat(a,
                 n,
                 precision=None):
    r"""Generates exponential correlation matrices

    This function computes for every element :math:`a` of a complex-valued
    tensor :math:`\mathbf{a}` the corresponding :math:`n\times n` exponential
    correlation matrix :math:`\mathbf{R}(a,n)`, defined as (Eq. 1, [MAL2018]_):

    .. math::
        \mathbf{R}(a,n)_{i,j} = \begin{cases}
                    1 & \text{if } i=j\\
                    a^{i-j}  & \text{if } i>j\\
                    (a^\star)^{j-i}  & \text{if } j<i, j=1,\dots,n\\
                  \end{cases}

    where :math:`|a|<1` and :math:`\mathbf{R}\in\mathbb{C}^{n\times n}`.

    Input
    -----
    a : [n_0, ..., n_k], `tf.complex`
        Parameters :math:`a` for the exponential correlation matrices

    n : `int`
        Number of dimensions of the output correlation matrices

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    R : [n_0, ..., n_k, n, n], `tf.complex`
        Correlation matrices
    """
    if precision is None:
        cdtype = config.tf_cdtype
    else:
        cdtype = dtypes[precision]["tf"]["cdtype"]

    # Cast to desired output dtype and expand last dimension for broadcasting
    a = tf.cast(a, dtype=cdtype)
    a = tf.expand_dims(a, -1)

    # Check that a is valid
    msg = "The absolute value of the elements of `a` must be smaller than one"
    tf.debugging.assert_less(tf.abs(a), tf.cast(1, a.dtype.real_dtype), msg)

    # Vector of exponents, adapt dtype and dimensions for broadcasting
    exp = tf.range(0, n)
    exp = tf.cast(exp, dtype=cdtype)
    exp = expand_to_rank(exp, tf.rank(a), 0)

    # First column of R
    col = tf.math.pow(a, exp)

    # For a=0, one needs to remove the resulting nans due to 0**0=nan
    cond = tf.math.is_nan(tf.math.real(col))
    col = tf.where(cond, tf.ones_like(col), col)

    # First row of R (equal to complex-conjugate of the first column)
    row = tf.math.conj(col)

    # Create Toeplitz operator
    operator = tf.linalg.LinearOperatorToeplitz(col, row)

    # Generate dense tensor from operator
    r = operator.to_dense()

    return r


def one_ring_corr_mat(phi_deg,
                      num_ant,
                      d_h=0.5,
                      sigma_phi_deg=15,
                      precision=None):
    r"""Generates covariance matrices from the one-ring model

    This function generates approximate covariance matrices for the
    so-called `one-ring` model (Eq. 2.24) [BHS2017]_. A uniform
    linear array (ULA) with uniform antenna spacing is assumed. The elements
    of the covariance matrices are computed as:

    .. math::
        \mathbf{R}_{\ell,m} =
              \exp\left( j2\pi d_\text{H} (\ell -m)\sin(\varphi) \right)
              \exp\left( -\frac{\sigma_\varphi^2}{2}
              \left( 2\pi d_\text{H}(\ell -m)\cos(\varphi) \right)^2 \right)

    for :math:`\ell,m = 1,\dots, M`, where :math:`M` is the number of antennas,
    :math:`\varphi` is the angle of arrival, :math:`d_\text{H}` is the antenna
    spacing in multiples of the wavelength,
    and :math:`\sigma^2_\varphi` is the angular standard deviation.

    Input
    -----
    phi_deg : [n_0, ..., n_k], `tf.float`
        Azimuth angles (deg) of arrival

    num_ant : `int`
        Number of antennas

    d_h : `float`, (default 0.5)
        Antenna spacing in multiples of the wavelength

    sigma_phi_deg : `float`, (default 15)
        Angular standard deviation (deg). Values greater
        than 15 should not be used as the approximation becomes invalid.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    R : [n_0, ..., n_k, num_ant, nun_ant], `tf.complex`
        Covariance matrices
    """
    if precision is None:
        rdtype = config.tf_rdtype
        cdtype = config.tf_cdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
        cdtype = dtypes[precision]["tf"]["cdtype"]

    if sigma_phi_deg > 15:
        warnings.warn("sigma_phi_deg should be smaller than 15.")

    # Convert all inputs to radians
    phi_deg = tf.cast(phi_deg, dtype=rdtype)
    sigma_phi_deg = tf.cast(sigma_phi_deg, dtype=rdtype)
    phi = deg_2_rad(phi_deg)
    sigma_phi = deg_2_rad(sigma_phi_deg)

    # Add dimensions for broadcasting
    phi = tf.expand_dims(phi, -1)
    sigma_phi = tf.expand_dims(sigma_phi, -1)

    # Compute first column
    c = tf.constant(2*PI*d_h, dtype=rdtype)
    d = c*tf.range(0, num_ant, dtype=rdtype)
    d = expand_to_rank(d, tf.rank(phi), 0)

    a = tf.complex(tf.cast(0, dtype=rdtype), d*tf.sin(phi))
    exp_a = tf.exp(a)  # First exponential term

    b = -tf.cast(0.5, dtype=rdtype)*(sigma_phi*d*tf.cos(phi))**2
    exp_b = tf.cast(tf.exp(b), dtype=cdtype)  # Second exponetial term

    col = exp_a*exp_b  # First column

    # First row is just the complex conjugate of first column
    row = tf.math.conj(col)

    # Create Toeplitz operator
    operator = tf.linalg.LinearOperatorToeplitz(col, row)

    # Generate dense tensor from operator
    r = operator.to_dense()

    return r
