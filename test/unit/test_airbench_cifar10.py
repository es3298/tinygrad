import unittest
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from examples.airbench_cifar10 import AirbenchMuon, NS_COEFFS, airbench_conv2d, random_permutation, whitening_covariance


class TestAirbenchCifar10(unittest.TestCase):
  def test_batched_muon_matches_individual_updates(self):
    Tensor.manual_seed(3)
    # The benchmark's BF16 Muon path targets NVIDIA GPUs; PYTHON makes this math-only test portable to CPU CI hosts.
    params = [Tensor.randn(2, 3, 1, 1, device="PYTHON").realize().is_param_(),
              Tensor.randn(3, 3, 1, 1, device="PYTHON").realize().is_param_()]
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

  def test_modular_shuffle_is_permutation(self):
    coefficients = Tensor([[1, 2, 3, 4], [4, 0, 6, 2], [3, 1, 5, 0], [2, 4, 1, 3]], dtype=dtypes.int32)
    self.assertEqual(sorted(random_permutation(5, 7, coefficients).tolist()), list(range(35)))

  def test_whitening_covariance_matches_numpy(self):
    Tensor.manual_seed(5)
    images = Tensor.randn(2, 3, 4, 4, dtype=dtypes.float32).realize()
    patches = np.lib.stride_tricks.sliding_window_view(images.numpy(), window_shape=(2, 2), axis=(2, 3))
    flat = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, 12)
    expected = (flat.T @ flat) / flat.shape[0]
    self.assertLess(float(np.abs(expected - whitening_covariance(images).numpy()).max()), 2e-6)

  def test_custom_conv_matches_native(self):
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


if __name__ == "__main__": unittest.main()
