import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json


# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.log(x)/tf.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# Model architecture
def roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw):
  """Converts roll-pitch-yaw angles to rotation matrices.
  Args:
    roll_pitch_yaw: Tensor (or convertible value) with shape [..., 3]. The last
      dimension contains the roll, pitch, and yaw angles in radians.  The
      resulting matrix rotates points by first applying roll around the x-axis,
      then pitch around the y-axis, then yaw around the z-axis.
  Returns:
     Tensor with shape [..., 3, 3]. The 3x3 rotation matrices corresponding to
     the input roll-pitch-yaw angles.
  """
  roll_pitch_yaw = tf.convert_to_tensor(roll_pitch_yaw)

  cosines = tf.cos(roll_pitch_yaw)
  sines = tf.sin(roll_pitch_yaw)
  cx, cy, cz = tf.unstack(cosines, axis=-1)
  sx, sy, sz = tf.unstack(sines, axis=-1)
  # pyformat: disable
  rotation = tf.stack(
      [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
       sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
       -sy, cy * sx, cy * cx], axis=-1)
  # pyformat: enable
  shape = tf.concat([tf.shape(rotation)[:-1], [3, 3]], axis=0)
  rotation = tf.reshape(rotation, shape)
  return rotation

def decode_covariance_roll_pitch_yaw(radii, rotations, invert=False):
  """Converts 6-D radus vectors to the corresponding covariance matrices.
  Args:
    radii: Tensor with shape [EC, 3]. Covariances of the three Gaussian axes. 
    rotations: Tensor with shape [EC, 3]. The roll-pitch-yaw rotation angles
        of the Gaussian frame.
    invert: Whether to return the inverse covariance.
  Returns:
     Tensor with shape [..., 3, 3]. The 3x3 (optionally inverted) covariance
     matrices corresponding to the input radius vectors.
  """
  DIV_EPSILON=1e-8
  d = 1.0 / (radii + DIV_EPSILON) if invert else radii
  diag = tf.matrix_diag(d)
  rotation = roll_pitch_yaw_to_rotation_matrices(rotations)
  return tf.matmul(tf.matmul(rotation, diag), rotation, transpose_b=True)

def eval_rbf(samples, centers, radii, rotations):
  """Samples gaussian radial basis functions at specified coordinates.
  Args:
    samples: Tensor with shape [N, 3], where N is the number of samples to evaluate.
    centers: Tensor with shape [EC, 3]. Contains the [x,y,z] coordinates of the
      RBF centers.
    radii: Tensor with shape [EC, 3]. First three numbers are covariances of
      the three Gaussian axes. 
    rotations: the roll-pitch-yaw rotation angles of the Gaussian frame.
  Returns:
     Tensor with shape [EC, N, 1]. The basis function strength at each sample.
        TODO(kgenova) maybe switch to [N, EC].
     location.
  """
  with tf.name_scope('sample_cov_bf'):
    assert len(samples.shape) == 2
    samples = tf.expand_dims(samples, axis=0) # Now shape is [1, N, 3]
    
    # Compute the samples' offset from center, then extract the coordinates.
    diff = samples - tf.expand_dims(centers, axis=-2) # broadcast to [1, n, 3] - [ec, 1, 3] -> [ec, n, 3]
    x, y, z = tf.unstack(diff, axis=-1)
    # Decode 6D radius vectors into inverse covariance matrices, then extract
    # unique elements.
    inv_cov = decode_covariance_roll_pitch_yaw(radii, rotations, invert=True)
    shape = tf.concat([tf.shape(inv_cov)[:-2], [1, 9]], axis=0)
    inv_cov = tf.reshape(inv_cov, shape)
    c00, c01, c02, _, c11, c12, _, _, c22 = tf.unstack(inv_cov, axis=-1)
    # Compute function value.
    dist = (
        x * (c00 * x + c01 * y + c02 * z) + y * (c01 * x + c11 * y + c12 * z) +
        z * (c02 * x + c12 * y + c22 * z))
    dist = tf.expand_dims(tf.exp(-0.5 * dist), axis=-1)
    #dist.set_shape([4, 65536, 1])
    return dist


def init_ldif(n_elts=32):
    #init_constants = np.random.uniform(low=2.0, high=4.0, size=(n_elts,1)).astype(np.float32)
    init_constants = np.random.uniform(low=100.0, high=200.0, size=(n_elts, 1)).astype(np.float32)
    #constant_vars = tf.Variable(initial_value=init_constants, trainable=True)
    constant_vars = tf.keras.backend.variable(value=init_constants, dtype='float32', name='constants')
    constants = tf.abs(constant_vars)  # Can't be negative

    init_centers = np.random.uniform(low=-1, high=1, size=(n_elts, 3)).astype(np.float32)
    centers = tf.Variable(
        initial_value=init_centers, 
        trainable=True)

    init_radii = np.random.uniform(low=0, high=0.3, size=(n_elts, 3)).astype(np.float32)
    radii_vars = tf.Variable(initial_value=init_radii, trainable=True)
    radii = tf.abs(radii_vars) # Can't be negative

    init_rotations = np.random.uniform(low=-1 * np.pi, high=np.pi, size=(n_elts, 3)).astype(np.float32)
    #rotation_vars = tf.Variable(initial_value=init_rotations, trainable=True)
    rotations = tf.keras.backend.variable(value=init_rotations, dtype='float32', name='rotations')
    #rotations = rotation_vars  # Wraps around since we just take sin/cos. Maybe need to do more here.
    # TODO(kgenova) When creating world2local frames, check 
    grad_vars = [constant_vars, centers, radii_vars, rotations]
    return (constants, centers, radii, rotations), grad_vars


class KFCLayer(tf.keras.layers.Layer):
    def __init__(self, n_duplicates, output_width, activation=None):
        super(KFCLayer, self).__init__()
        self.output_width = output_width
        self.n_duplicates = n_duplicates
        self.activation = activation

    def build(self, input_shape):
        print('Input shape: ', input_shape)
        input_width = input_shape[-1]
        initializer = tf.keras.initializers.glorot_uniform()
        w_inits = np.stack([initializer(shape=(input_width, self.output_width), dtype='float32') for _ in range(self.n_duplicates)])
        #print(f'Initial weights: {w_inits}')
        #w_init = tf.keras.initializers.GlorotUniform()(shape=(input_width, output_width))
        self.w = tf.Variable(initial_value=w_inits, trainable=True, name='BatchWeight', dtype=tf.float32)
        self.b = self.add_weight(shape=(self.n_duplicates, 1, self.output_width), initializer='zeros', trainable=True, name='BatchBias', dtype=tf.float32)

        print(f'Weights, bias shape: {self.w.shape} and {self.b.shape}')
        #import sys
        #sys.exit(1)
        #self.weights = self.add_weight(shape=(self.n_duplicates, input_width, output_width

    def call(self, inputs):
        #print(f'KFCLayer input shape: {inputs.shape}')
        assert len(inputs.shape) == 3
        assert inputs.shape[0] == self.n_duplicates
        # inputs = tf.expand_dims(inputs, axis=0)
        # inputs = tf.tile(inputs, [self.n_duplicates, 1, 1])
        outputs = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            outputs = self.activation(outputs)
        #print(f'KFCLayer output shape: {outputs.shape}')
        return outputs


class LDIFLayer(tf.keras.layers.Layer):
    def __init__(self, n_elts):
        super(LDIFLayer, self).__init__()
        self.n_elts = n_elts

    def build(self, input_shape):
        self.constants = self.add_weight(name='constants', shape=(self.n_elts, 1),
                initializer=tf.keras.initializers.RandomUniform(10.0, 20.0),
                trainable=True)
        self.centers = self.add_weight(name='centers', shape=(self.n_elts, 3),
                initializer=tf.keras.initializers.RandomUniform(-0.45, 0.45),
                trainable=True)
        self.radii = self.add_weight(name='radii', shape=(self.n_elts, 1),
                initializer=tf.keras.initializers.RandomUniform(0.05, 0.06),
                trainable=True)
        self.rotations = self.add_weight(name='rotations', shape=(self.n_elts, 3),
                initializer=tf.keras.initializers.RandomUniform(-1.0 * np.pi, np.pi),
                trainable=True)
        self.call_count = 0

    def call(self, world_space_points, nerflet_activations, dists):
        constants = tf.abs(self.constants)
        centers = self.centers
        radii = tf.tile(tf.abs(self.radii), [1, 3])
        rotations = self.rotations  # Wraps around infinitely?
        
        # Inputs are NeRF outputs to be blended:
        rbfs = eval_rbf(world_space_points, centers, radii, rotations)
        constants = tf.expand_dims(constants, axis=1)
        rbfs = rbfs * constants

        # Apply an l1 penalty so that blobs don't overlap much in influence:
        no_penalty_threshold = 0.005
        penalty = tf.reduce_mean(tf.reduce_sum(tf.abs(rbfs - no_penalty_threshold), axis=0))


        assert len(rbfs.shape) == 3

        #assert len(nerflet_activations) == self.n_elts
        if isinstance(nerflet_activations, list):
            nerflet_activations = tf.stack(nerflet_activations)

        # Abs because each nerflet should be nonegative in rgb and opacity everywhere:
        #print(f'Nerflet activation shape: {nerflet_activations.shape}')
        #nerflet_activations = tf.stack(nerflet_activations)

        # Map to RGBa:
        alpha = 1.0 - tf.exp(-tf.nn.relu(nerflet_activations[..., -1:]) * tf.expand_dims(dists, axis=0))
        rgb = tf.math.sigmoid(nerflet_activations[..., :3])
        nerflet_activations = tf.concat([rgb, alpha], axis=-1)

        #nerflet_activations = tf.keras.activations.sigmoid(tf.stack(nerflet_activations))
        #nerflet_activations = tf.concat([nerflet_activations[..., :3], tf.abs(nerflet_act

        n_to_print = 5


        #min_weight = 0.01
        rbf_sums = tf.reduce_sum(rbfs, axis=0, keepdims=True) # [1, N, 1]
        #rbf_sums = tf.maximum(rbf_sums, min_weight)
        rbfs = rbfs / (rbf_sums + 0.01)
        outputs = tf.reduce_sum(nerflet_activations * rbfs, axis=0)

        if self.call_count % 10 == 0:
            tf.print('First points: ', world_space_points[:n_to_print, :])
            tf.print('First rbf values: ', rbfs[:, :n_to_print, :])
            tf.print('First nerflet activations: ', nerflet_activations[:, :n_to_print, :])
            tf.print('First final results: ', outputs[:n_to_print, :])
            tf.print('Average RBF sum: ', tf.reduce_mean(rbf_sums))
            tf.print('Average (multiplied) rbf weight: ', tf.reduce_mean(rbfs))
            # Compute the average number of nontrivial weights per nerf (and the mean per cell):
            for thresh in [0.5, 0.1, 0.05, 0.01, 0.005, 0.0001]:
                rbf_nontrivial = tf.cast(rbfs > thresh, dtype=tf.float32)
                n_nontrivial = tf.reduce_mean(tf.reduce_sum(rbf_nontrivial, axis=0))
                tf.print(f'Average number of nontrivial rbfs per point at thresh {thresh}: ', n_nontrivial)
                n_nontrivial = tf.reshape(tf.reduce_sum(rbf_nontrivial, axis=1), [-1])
                tf.print(f'Number of nontrivial points per rbf at thresh {thresh}: ', n_nontrivial)
                print(f'Shape of nontrivial points per rbf: {n_nontrivial.shape}')
                frac_sums_nontrivial = tf.reduce_mean(tf.cast(rbf_sums > thresh, dtype=tf.float32))
                tf.print(f'Fraction of RBF sums that are nontrivial at thresh {thresh}: ', frac_sums_nontrivial)

        self.call_count += 1
        return outputs, penalty


def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,
    n_elts=None):
    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)
    def stacked_dense(W, act=relu): return KFCLayer(n_elts, W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    # TODO(kgenova) If input channels is 3, when does the embedding get applied?
    #assert input_ch == 63
    #assert input_ch_views == 2
    assert output_ch == 4
    assert use_viewdirs
    assert n_elts is not None
    #constants, centers, radii, rotations = sif_params

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views + 3 + 1))
    inputs_pts, inputs_views, input_unembedded_pts, dists = tf.split(inputs, [input_ch, input_ch_views, 3, 1], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])
    input_unembedded_pts.set_shape([None, 3])

    stacked_inputs_pts = tf.tile(tf.expand_dims(inputs_pts, 0), [n_elts, 1, 1])
    stacked_inputs_views = tf.tile(tf.expand_dims(inputs_views, 0), [n_elts, 1, 1])
    stacked_input_unembedded_pts = tf.tile(tf.expand_dims(input_unembedded_pts, 0), [n_elts, 1, 1])

    #rbfs = eval_rbf(input_unembedded_pts, centers, radii, rotations)  # Shape [EC, N, 1]
    #print(f'RBF Shapes: {rbfs.shape}')
    #constants = tf.expand_dims(constants, axis=1)  # Shape [EC, 1, 1]
    #rbfs = rbfs * constants  # TODO(kgenova) If normalizing, need to scale so that weight sum is usually > 1.0
    # TODO(kgenova) Since background is white, maybe we should decay to [1, 1, 1, 0], not [0, 0, 0, 0].
    # TODO(kgenova) An alternate way of avoiding degeneracy is to have a dummy blob with weight X that wants [1, 1, 1, 0].
    #assert len(rbfs.shape) == 3
    # TODO(kgenova) Make these loosely sum up to one?
    #min_weight = 1.0
    #rbf_sums = tf.reduce_sum(rbfs, axis=0, keepdims=True) # [1, N, 1]
    #rbf_sums = tf.maximum(rbf_sums, min_weight)
    #rbfs = rbfs / rbf_sums
    #rbfs = tf.keras.backend.print_tensor(rbfs, message='rbfs = ')
    #tf.print("RBFs: ", rbfs)

    #rbfs = tf.zeros_like(rbfs)

    # V0 (commented out):
    if False:
        nerflet_activations = []
        for i in range(n_elts):
          # Make a NeRF per element:
          print(inputs.shape, inputs_pts.shape, inputs_views.shape)
          # TODO(kgenova) Consider transforming (pre-embedding) to local frame, instead of doing all in global frame.
          # Need to also transform input views for that...
          outputs = inputs_pts
          for i in range(D):
              outputs = dense(W)(outputs)
              if i in skips:
                  outputs = tf.concat([inputs_pts, outputs], -1)

          if use_viewdirs:
              alpha_out = dense(1, act=None)(outputs)
              bottleneck = dense(W, act=None)(outputs)
              inputs_viewdirs = tf.concat(
                  [bottleneck, inputs_views], -1)  # concat viewdirs
              outputs = inputs_viewdirs
              # The supplement to the paper states there are 4 hidden layers here, but this is an error since
              # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
              for i in range(1):
                  outputs = dense(W//2)(outputs)
              outputs = dense(3, act=None)(outputs)
              outputs = tf.concat([outputs, alpha_out], -1)
          else:
              outputs = dense(output_ch, act=None)(outputs)
          nerflet_activations.append(outputs)
    else:
        outputs = stacked_inputs_pts
        for i in range(D):
            outputs = stacked_dense(W)(outputs)
            if i in skips:
                outputs = tf.concat([stacked_inputs_pts, outputs], -1)
        if use_viewdirs:
            alpha_out = stacked_dense(1, act=None)(outputs)
            bottleneck = stacked_dense(W, act=None)(outputs)
            inputs_viewdirs = tf.concat(
                    [bottleneck, stacked_inputs_views], -1)
            outputs = inputs_viewdirs
            for i in range(1):
                outputs = stacked_dense(W//2)(outputs)
            outputs = stacked_dense(3, act=None)(outputs)
            outputs = tf.concat([outputs, alpha_out], -1)
        nerflet_activations = outputs
   
    #assert len(outputs_per_elt) == n_elts
    #outputs = tf.stack(outputs_per_elt, axis=0)  # Now should have shape [EC, N, output_ch]
    #assert len(outputs.shape) == 3
    #outputs = tf.reduce_sum(outputs * rbfs, axis=0)
    # outputs should have shape [N, output_ch]
    ldif = LDIFLayer(n_elts=n_elts)
    outputs = ldif(input_unembedded_pts, nerflet_activations, dists)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Ray helpers

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples
