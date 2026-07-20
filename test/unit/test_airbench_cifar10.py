import contextlib, io, sys, unittest
from unittest.mock import patch
import numpy as np

from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import Context
from examples.airbench_cifar10 import (AirbenchBatchNorm, AirbenchCifarNet, AirbenchMuon, NS_COEFFS, airbench_conv2d, main,
                                       permutation_parameters, random_permutation, dirac_init_, symmetric_eigh, whitening_covariance,
                                       whitening_patch_mean)


class TestAirbenchCifar10(unittest.TestCase):
  def test_batched_muon_matches_individual_updates(self):
    Tensor.manual_seed(3)
    # The benchmark's BF16 Muon path targets NVIDIA GPUs; PYTHON makes this math-only test portable to CPU CI hosts.
    params = [Tensor.randn(2, 3, 1, 1, device="PYTHON").realize().is_param_(),
              Tensor.randn(2, 3, 1, 1, device="PYTHON").realize().is_param_()]
    reference = [param.clone().realize() for param in params]
    reference_momentum = [Tensor.zeros_like(param).realize() for param in params]
    lr, momentum, weight_decay = 0.1, 0.6, 0.02
    optimizer = AirbenchMuon(params, lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=3)

    for normalize_weights in (False, True):
      grads = [Tensor.randn(*param.shape, device="PYTHON").realize() for param in params]
      expected = []
      for param, buffer, grad in zip(reference, reference_momentum, grads):
        buffer.assign(momentum * buffer + grad).realize()
        update = (grad + momentum * buffer).reshape(grad.shape[0], -1).cast(dtypes.bfloat16).newton_schulz(3, NS_COEFFS)
        base = param * ((param.shape[0] ** 0.5) / (param.float().square().sum().sqrt() + 1e-7)) if normalize_weights else param
        expected.append(((base - lr * update.reshape(grad.shape).cast(param.dtype)) * (1.0 - lr * weight_decay)).realize())
      for param, grad in zip(params, grads): param.grad = grad
      optimizer.normalize_weights = normalize_weights
      with Context(TRAINING=1): Tensor.realize(*optimizer.schedule_step())
      for actual, target in zip(params, expected): np.testing.assert_allclose(actual.numpy(), target.numpy(), atol=2e-5, rtol=2e-5)
      reference = expected

  def test_muon_first_step_matches_numpy(self):
    param_np = np.array([[[[0.2]], [[-0.3]], [[0.5]]], [[[0.7]], [[0.1]], [[-0.4]]]], dtype=np.float32)
    grad_np = np.array([[[[0.4]], [[-0.2]], [[0.1]]], [[[-0.5]], [[0.3]], [[0.6]]]], dtype=np.float32)
    lr, momentum, weight_decay = 0.1, 0.6, 0.02
    param = Tensor(param_np, device="PYTHON").realize().is_param_()
    param.grad = Tensor(grad_np, device="PYTHON").realize()
    optimizer = AirbenchMuon([param], lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=3)
    optimizer.normalize_weights = False
    with Context(TRAINING=1): Tensor.realize(*optimizer.schedule_step())

    update = grad_np * (1.0 + momentum)
    matrix = update.reshape(update.shape[0], -1)
    matrix = matrix / (np.sqrt(np.square(matrix).sum()) + 1e-7)
    a, b, c = NS_COEFFS
    for _ in range(3):
      gram = matrix @ matrix.T
      matrix = a * matrix + (b * gram + c * (gram @ gram)) @ matrix
    expected = (param_np - lr * matrix.reshape(param_np.shape)) * (1.0 - lr * weight_decay)
    np.testing.assert_allclose(param.numpy(), expected, atol=8e-3, rtol=8e-3)

  def test_muon_tensor_normalization_mask_matches_boolean(self):
    Tensor.manual_seed(13)
    for normalize in (False, True):
      initial = Tensor.randn(2, 3, 1, 1, device="PYTHON").realize()
      grad = Tensor.randn(*initial.shape, device="PYTHON").realize()
      params = [initial.clone().is_param_(), initial.clone().is_param_()]
      optimizers = [AirbenchMuon([param], lr=0.1, momentum=0.6, ns_steps=3) for param in params]
      optimizers[0].normalize_weights = normalize
      optimizers[1].normalize_weights = Tensor(normalize, dtype=dtypes.bool, device="PYTHON").realize()
      for param, optimizer in zip(params, optimizers):
        param.grad = grad.clone().realize()
        with Context(TRAINING=1): Tensor.realize(*optimizer.schedule_step())
      np.testing.assert_allclose(params[0].numpy(), params[1].numpy(), atol=2e-5, rtol=2e-5)

  def test_batchnorm_keeps_fixed_scale_out_of_autograd(self):
    default_float = dtypes.default_float
    try:
      dtypes.default_float = dtypes.half
      norm = AirbenchBatchNorm(3, eps=1e-12, momentum=0.4)
      x = Tensor.randn(2, 3, 4, 4, dtype=dtypes.half, device=norm.bias.device).realize().is_param_()
      with Context(TRAINING=1): norm(x).sum().backward()
      Tensor.realize(x.grad, norm.bias.grad, norm.running_mean, norm.running_var)
      self.assertIsNone(norm.weight)
      self.assertEqual(norm.running_mean.dtype, dtypes.float32)
      self.assertEqual(norm.running_var.dtype, dtypes.float32)
      self.assertIsNotNone(norm.bias.grad)
    finally: dtypes.default_float = default_float

  def test_whitening_parameters_are_frozen(self):
    model = AirbenchCifarNet(bn_eps=1e-12, bn_momentum=0.4, width=16)
    self.assertFalse(model.whiten.weight.is_param)
    self.assertFalse(model.whiten.bias.is_param)

  def test_zero_steps_are_rejected_before_setup(self):
    with contextlib.redirect_stderr(io.StringIO()), patch.object(sys, "argv", ["airbench_cifar10.py", "--steps", "0"]), \
         self.assertRaises(SystemExit):
      main()

  def test_modular_shuffle_is_permutation(self):
    coefficients = Tensor([[1, 2, 3, 4], [4, 0, 6, 2], [3, 1, 5, 0], [2, 4, 1, 3]], dtype=dtypes.int32)
    self.assertEqual(sorted(random_permutation(5, 7, coefficients).tolist()), list(range(35)))

  def test_permutation_parameters_are_deterministic_and_bounded(self):
    first, second = permutation_parameters(2, 2, 5, 7), permutation_parameters(2, 2, 5, 7)
    self.assertEqual(first.tolist(), second.tolist())
    for values in first.tolist():
      self.assertTrue(all(0 <= row[0] < 5 and 0 <= row[1] < 5 for row in values))
      self.assertTrue(all(0 <= row[2] < 7 and 0 <= row[3] < 7 for row in values))

  def test_symmetric_eigh_reconstructs_matrix(self):
    matrix = np.array([[4.0, 1.0, -0.5], [1.0, 2.0, 0.25], [-0.5, 0.25, 1.0]], dtype=np.float64)
    values, vectors = symmetric_eigh(matrix.tolist())
    vectors_np = np.array(vectors)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(matrix), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(vectors_np.T @ vectors_np, np.eye(3), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(vectors_np @ np.diag(values) @ vectors_np.T, matrix, atol=1e-10, rtol=1e-10)

  def test_whitening_covariance_matches_numpy(self):
    Tensor.manual_seed(5)
    images = Tensor.randn(2, 3, 4, 4, dtype=dtypes.float32).realize()
    patches = np.lib.stride_tricks.sliding_window_view(images.numpy(), window_shape=(2, 2), axis=(2, 3))
    flat = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, 12)
    expected = (flat.T @ flat) / flat.shape[0]
    self.assertLess(float(np.abs(expected - whitening_covariance(images).numpy()).max()), 2e-6)
    np.testing.assert_allclose(whitening_patch_mean(images).numpy(), flat.mean(axis=0), atol=2e-6, rtol=2e-6)

  def test_airbench_conv_matches_native(self):
    Tensor.manual_seed(11)
    x = Tensor.randn(2, 3, 5, 5, dtype=dtypes.float32).realize()
    weight = Tensor.randn(4, 3, 3, 3, dtype=dtypes.float32).realize()
    grad = Tensor.randn(2, 4, 5, 5, dtype=dtypes.float32).realize()
    x_native, weight_native = x.clone().is_param_(), weight.clone().is_param_()
    x_custom, weight_custom = x.clone().is_param_(), weight.clone().is_param_()

    native = x_native.conv2d(weight_native, padding=1)
    custom = airbench_conv2d(x_custom, weight_custom)
    (native * grad).sum().backward()
    (custom * grad).sum().backward()

    self.assertLess(float((native - custom).abs().max().item()), 2e-6)
    self.assertLess(float((x_native.grad - x_custom.grad).abs().max().item()), 3e-6)
    self.assertLess(float((weight_native.grad - weight_custom.grad).abs().max().item()), 2e-6)

  def test_dirac_init_preserves_expansion_channels(self):
    Tensor.manual_seed(7)
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False)
    before = conv.weight.clone().realize()
    dirac_init_(conv)
    after = conv.weight.realize().numpy()
    expected = np.zeros((3, 3, 3, 3), dtype=np.float32)
    expected[np.arange(3), np.arange(3), 1, 1] = 1
    np.testing.assert_array_equal(after[:3], expected)
    np.testing.assert_array_equal(after[3:], before.numpy()[3:])


if __name__ == "__main__": unittest.main()
