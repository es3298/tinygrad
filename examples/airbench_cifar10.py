#!/usr/bin/env python3
"""Train CIFAR-10 to 94% with a tinygrad-tuned Airbench/Hiverge recipe.

TinyJit compilation is warmed before the timed region. Whitening, training, evaluation, and accuracy readback are timed.
The recipe is derived from MIT-licensed Airbench and Hiverge code, then retuned for tinygrad's NV backend on one L40S:
https://github.com/KellerJordan/cifar10-airbench/tree/4c1b6d1e3889b037efadcfd5c0ea65b246592362
https://github.com/hiverge/cifar10-speedrun/tree/06c60727547042c847d919c5848e807e8119d582

Run on one NVIDIA GPU with: DEV=NV JITBEAM=4 python3 examples/airbench_cifar10.py
"""
import argparse, math, os, subprocess, time
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, Device, dtypes, nn, Variable
from tinygrad.helpers import Context, getenv, TRAINING
from tinygrad.uop.ops import KernelInfo

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NS_COEFFS = (3.4576, -4.7391, 2.0843)
WIDTH, BN_EPS, BN_MOMENTUM, LOGIT_DIV = 256, 1e-12, 0.4434, 256.0
BIAS_LR, HEAD_LR, MUON_LR = 0.0573, 0.5415, 0.205
SGD_MOMENTUM, MUON_MOMENTUM, WEIGHT_DECAY = 0.825, 0.655, 1.0418e-6
LABEL_SMOOTHING, BRIGHTNESS, CONTRAST = 0.09, 0.1399, 0.1308
TTA_BASE_WEIGHT = 0.2

def whiten_activ(x:Tensor) -> Tensor:
  return x.gelu("none")

def block_activ(x:Tensor) -> Tensor:
  return x.silu()

def activation_buffer(x:Tensor) -> Tensor:
  return x.contiguous()

def _conv_weight_grad(x:Tensor, dy:Tensor) -> Tensor:
  n, cin, h, w = x.shape
  cout = dy.shape[1]
  patches = x.pad((1, 1, 1, 1))._pool((3, 3)).permute(0, 2, 3, 1, 4, 5).reshape(n * h * w, cin * 9)
  dy_matrix = dy.permute(0, 2, 3, 1).reshape(n * h * w, cout).contiguous()
  return dy_matrix.T.matmul(patches).reshape(cout, cin, 3, 3)

def _conv_input_grad(dy:Tensor, weight:Tensor) -> Tensor:
  n, cout, h, w = dy.shape
  cin = weight.shape[1]
  dy_matrix = dy.permute(0, 2, 3, 1).reshape(n * h * w, cout).contiguous()
  dpatches = dy_matrix.matmul(weight.reshape(cout, cin * 9).contiguous()).reshape(n, h, w, cin, 3, 3)
  parts = []
  for ky in range(3):
    for kx in range(3):
      part = dpatches[:, :, :, :, ky, kx].permute(0, 3, 1, 2)
      parts.append(part.pad((kx, 2-kx, ky, 2-ky))[:, :, 1:-1, 1:-1])
  return Tensor.stack(*parts).sum(axis=0)

def _conv_gradient_hook(_raw:UOp, _x:UOp, _weight:UOp) -> UOp:
  # custom_kernel's AFTER edge is tinygrad's supported way to attach a custom gradient
  # while retaining the scheduler-optimized forward matmul.
  return UOp.sink(arg=KernelInfo(name="airbench_conv_gradient_hook"))

def airbench_conv2d(x:Tensor, weight:Tensor) -> Tensor:
  """3x3 padded convolution with GEMM-shaped gradients for NVIDIA tensor cores."""
  n, cin, h, w = x.shape
  cout = weight.shape[0]
  patches = x.detach().pad((1, 1, 1, 1))._pool((3, 3)).permute(0, 2, 3, 1, 4, 5).reshape(n * h * w, cin * 9)
  raw = patches.contiguous().matmul(weight.detach().reshape(cout, cin * 9).T)
  def backward(grad:UOp, call:UOp) -> tuple[None, UOp, UOp]:
    _raw, call_x, call_weight = call.src[1:]
    dy = Tensor(grad).cast(call_x.dtype).reshape(n, h, w, cout).permute(0, 3, 1, 2)
    return None, _conv_input_grad(dy, Tensor(call_weight)).uop, _conv_weight_grad(Tensor(call_x), dy).uop
  raw = Tensor.custom_kernel(raw, x, weight, fxn=_conv_gradient_hook, grad_fxn=backward)[0]
  return raw.reshape(n, h, w, cout).permute(0, 3, 1, 2)

class AirbenchBatchNorm(nn.BatchNorm2d):
  def __init__(self, channels:int, eps:float, momentum:float):
    super().__init__(channels, eps=eps, momentum=momentum, affine=False, track_running_stats=True)
    self.bias = Tensor.zeros(channels, dtype=dtypes.float32)
    self.running_mean = Tensor.zeros(channels, dtype=dtypes.float32).is_param_(False)
    self.running_var = Tensor.ones(channels, dtype=dtypes.float32).is_param_(False)

class Conv:
  def __init__(self, channels_in:int, channels_out:int):
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False)
    dirac_init_(self.conv)

  def __call__(self, x:Tensor) -> Tensor:
    return airbench_conv2d(x, self.conv.weight)

class ConvGroup:
  def __init__(self, channels_in:int, channels_out:int, bn_eps:float, bn_momentum:float):
    self.conv1, self.conv2 = Conv(channels_in, channels_out), Conv(channels_out, channels_out)
    self.norm1 = AirbenchBatchNorm(channels_out, bn_eps, bn_momentum)
    self.norm2 = AirbenchBatchNorm(channels_out, bn_eps, bn_momentum)

  def __call__(self, x:Tensor) -> Tensor:
    x = activation_buffer(self.conv1(x)).max_pool2d(2).float()
    x = activation_buffer(block_activ(self.norm1(x).cast(dtypes.default_float)))
    x = activation_buffer(self.conv2(x)).float()
    return activation_buffer(block_activ(self.norm2(x).cast(dtypes.default_float)))

class AirbenchCifarNet:
  def __init__(self, bn_eps:float, bn_momentum:float, logit_div:float, width:int=256):
    self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
    self.whiten.weight.is_param_(False)
    self.block1 = ConvGroup(24, 64, bn_eps, bn_momentum)
    self.block2 = ConvGroup(64, width, bn_eps, bn_momentum)
    self.block3 = ConvGroup(width, width, bn_eps, bn_momentum)
    self.head = nn.Linear(width, 10, bias=False)
    self.logit_div = logit_div
    self.head.weight.assign((self.head.weight / self.head.weight.float().std()).cast(self.head.weight.dtype))

  def __call__(self, x:Tensor, whiten_bias_grad=True) -> Tensor:
    bias = self.whiten.bias if whiten_bias_grad else self.whiten.bias.detach()
    x = activation_buffer(whiten_activ(x.conv2d(self.whiten.weight.detach(), bias)))
    x = activation_buffer(self.block1(x))
    x = activation_buffer(self.block2(x))
    x = activation_buffer(self.block3(x))
    x = x.max_pool2d(3).reshape(x.shape[0], -1)
    return self.head(x) / self.logit_div

class AirbenchMuon(nn.optim.Optimizer):
  def __init__(self, params:list[Tensor], lr=0.205, momentum=0.655, weight_decay=0.0, ns_steps=3):
    super().__init__(params, lr, fused=False)
    self.momentum, self.weight_decay, self.ns_steps = momentum, weight_decay, ns_steps
    self.normalize_weights = False
    self.b = self._new_optim_param()

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    momentum_grads = []
    for i, (p, g) in enumerate(zip(params, grads)):
      g = g.cast(self.b[i].dtype)
      self.b[i].assign(self.momentum * self.b[i] + g)
      momentum_grads.append(g + self.momentum * self.b[i])

    rows = [g.shape[0] for g in momentum_grads]
    cols = [g.numel() // g.shape[0] for g in momentum_grads]
    max_rows, max_cols = max(rows), max(cols)
    padded = [g.reshape(r, c).pad(((0, max_rows-r), (0, max_cols-c))) for g, r, c in zip(momentum_grads, rows, cols)]
    orthogonal = Tensor.stack(*padded).cast(dtypes.bfloat16).newton_schulz(self.ns_steps, NS_COEFFS)
    momentum_grads = [orthogonal[i, :r, :c].reshape(g.shape) for i, (g, r, c) in enumerate(zip(momentum_grads, rows, cols))]

    updates = []
    for p, g in zip(params, momentum_grads):
      base = p.detach()
      if self.normalize_weights: base = base * ((p.shape[0] ** 0.5) / (base.float().square().sum().sqrt() + 1e-7))
      updated = (base - self.lr * g.cast(p.dtype)) * (1.0 - self.lr * self.weight_decay)
      updates.append((p.detach() - updated).cast(p.dtype))
    return updates, self.b

def dirac_init_(conv:nn.Conv2d) -> None:
  # Match Airbench: preserve random expansion channels, Dirac only the first input-width block.
  w = conv.weight.float().numpy().astype(np.float32)
  oc, ic, kh, kw = w.shape
  n = min(oc, ic)
  w[:n] = 0.0
  for i in range(n): w[i, i, kh//2, kw//2] = 1.0
  conv.weight.assign(Tensor(w, dtype=dtypes.float32).cast(conv.weight.dtype))

def pad_reflect(X:Tensor, size=2) -> Tensor:
  X = X[...,:,1:size+1].flip(-1).cat(X, X[...,:,-(size+1):-1].flip(-1), dim=-1)
  X = X[...,1:size+1,:].flip(-2).cat(X, X[...,-(size+1):-1,:].flip(-2), dim=-2)
  return X

def random_crop_batch(X:Tensor, batch_idxs:Tensor, crop_size=32) -> Tensor:
  batch_size, channels, height, width = batch_idxs.shape[0], X.shape[1], X.shape[2], X.shape[3]
  low_x = Tensor.randint(batch_size, low=0, high=width-crop_size+1).reshape(batch_size, 1, 1, 1)
  low_y = Tensor.randint(batch_size, low=0, high=height-crop_size+1).reshape(batch_size, 1, 1, 1)
  image = batch_idxs.reshape(batch_size, 1, 1, 1) * (channels * height * width)
  channel = Tensor.arange(channels, dtype=dtypes.int32).reshape(1, channels, 1, 1) * (height * width)
  y = (low_y + Tensor.arange(crop_size, dtype=dtypes.int32).reshape(1, 1, crop_size, 1)) * width
  x = low_x + Tensor.arange(crop_size, dtype=dtypes.int32).reshape(1, 1, 1, crop_size)
  return X.reshape(-1)[image + channel + y + x]

def random_permutation(rows:int, cols:int, coefficients:Tensor) -> Tensor:
  """Build a bijection over a rows x cols grid using alternating modular shears."""
  idx = Tensor.arange(rows * cols, dtype=dtypes.int32)
  row, col = idx // cols, idx % cols
  for round_idx in range(coefficients.shape[0]):
    row = (row + col * col + coefficients[round_idx, 0] * col + coefficients[round_idx, 1]) % rows
    col = (col + row * row + coefficients[round_idx, 2] * row + coefficients[round_idx, 3]) % cols
  return row * cols + col

def batch_color_jitter(X:Tensor, brightness:float, contrast:float) -> Tensor:
  shape = (X.shape[0], 1, 1, 1)
  brightness_shift = (Tensor.rand(*shape, dtype=X.dtype) * 2.0 - 1.0) * brightness
  contrast_scale = (Tensor.rand(*shape, dtype=X.dtype) * 2.0 - 1.0) * contrast + 1.0
  return (X + brightness_shift) * contrast_scale

def batch_random_flip(X:Tensor) -> Tensor:
  return (Tensor.rand(X.shape[0], 1, 1, 1) < 0.5).where(X.flip(-1), X)

@TinyJit
@Context(ALLOW_TF32=0)
def whitening_covariance(images:Tensor) -> Tensor:
  patches = images.float()._pool((2, 2)).permute(0, 2, 3, 1, 4, 5).reshape(-1, 12)
  return ((patches.T @ patches) / patches.shape[0]).realize()

def init_whitening_(model:AirbenchCifarNet, train_images:Tensor, eps=5e-4, n=5000) -> None:
  images = train_images[:n].contiguous().realize()
  cov = whitening_covariance(images).numpy()
  vals, vecs = np.linalg.eigh(cov, UPLO="U")
  w12 = (vecs.T.reshape(-1, 3, 2, 2) / np.sqrt(vals.reshape(-1,1,1,1) + eps)).astype(np.float32)
  w = np.concatenate((w12, -w12), axis=0)
  model.whiten.weight.assign(Tensor(w, dtype=dtypes.float32).cast(dtypes.default_float)).realize()

def preprocess_cifar() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
  X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
  mean = Tensor(CIFAR_MEAN, dtype=dtypes.default_float).reshape(1,3,1,1)
  std = Tensor(CIFAR_STD, dtype=dtypes.default_float).reshape(1,3,1,1)
  scale = Tensor(255.0, dtype=dtypes.default_float)
  X_train = ((X_train.cast(dtypes.default_float) / scale - mean) / std).realize()
  X_test = activation_buffer((X_test.cast(dtypes.default_float) / scale - mean) / std).realize()
  X_flip = batch_random_flip(X_train).realize()
  return pad_reflect(X_flip, 2).realize(), pad_reflect(X_flip.flip(-1), 2).realize(), X_train, X_test.realize(), Y_train.realize(), Y_test.realize()

def cross_entropy_sum(logits:Tensor, labels:Tensor, label_smoothing:float) -> Tensor:
  return logits.float().sparse_categorical_crossentropy(labels, reduction="sum", label_smoothing=label_smoothing)

def select_tta_indices(logits:Tensor, count:int) -> Tensor:
  top_two, _ = logits.topk(2)
  confidence = top_two[:, 0] - top_two[:, 1]
  return confidence.topk(count, largest=False)[1]

def main() -> None:
  parser = argparse.ArgumentParser(description="Airbench/Hiverge CIFAR-10 speed benchmark in tinygrad")
  parser.add_argument("--batch-size", type=int, default=getenv("BS", 2000))
  parser.add_argument("--steps", type=int, default=getenv("STEPS", 200))
  parser.add_argument("--eval-batch-size", type=int, default=getenv("EVAL_BS", 2000))
  parser.add_argument("--tta-level", type=int, choices=(0, 1, 2), default=getenv("TTA_LEVEL", 2))
  parser.add_argument("--tta-fraction", type=float, default=getenv("TTA_FRACTION", 0.18))
  parser.add_argument("--seed", type=int, default=getenv("SEED", 2))
  parser.add_argument("--target-acc", type=float, default=getenv("TARGET_EVAL_ACC_PCT", 94.0))
  parser.add_argument("--target-time", type=float, default=getenv("TARGET_TIME_S", 10.0))
  parser.add_argument("--quiet", action="store_true", default=bool(getenv("QUIET", 0)))
  parser.add_argument("--profile-phases", action="store_true", default=bool(getenv("PROFILE_PHASES", 0)))
  parser.add_argument("--no-warmup", action="store_true", help="include first-time JIT compilation in wall time")
  parser.add_argument("--save-weights", type=str, default="")
  args = parser.parse_args()
  if args.batch_size <= 0: parser.error("--batch-size must be positive")
  if args.steps <= 0: parser.error("--steps must be positive")
  if args.eval_batch_size <= 0: parser.error("--eval-batch-size must be positive")
  if not 0.0 < args.tta_fraction <= 1.0: parser.error("--tta-fraction must be greater than 0 and at most 1")

  phase_start = time.perf_counter()
  def phase(name:str) -> None:
    nonlocal phase_start
    if args.profile_phases and not args.quiet:
      now = time.perf_counter()
      print(f"phase={name} seconds={now-phase_start:.4f}", flush=True)
      phase_start = now

  total_start = time.perf_counter()
  Tensor.manual_seed(args.seed)
  dtypes.default_float = dtypes.half
  model = AirbenchCifarNet(BN_EPS, BN_MOMENTUM, LOGIT_DIV, WIDTH)
  phase("model_init")
  X_train_pad, X_train_pad_flip, X_train_norm, X_test, Y_train, Y_test = preprocess_cifar()
  phase("data_preprocess")

  state = nn.state.get_state_dict(model)
  groups = (model.block1, model.block2, model.block3)
  hidden_convs = [conv.conv.weight for group in groups for conv in (group.conv1, group.conv2)]
  norm_biases = [norm.bias for group in groups for norm in (group.norm1, group.norm2)]
  whiten_bias = [model.whiten.bias]
  head = [model.head.weight]
  bn_buffers = [v for k,v in state.items() if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k]

  batch_size, steps = args.batch_size, args.steps
  if X_train_norm.shape[0] % batch_size: parser.error("--batch-size must divide the CIFAR-10 training set")
  if X_test.shape[0] % args.eval_batch_size: parser.error("--eval-batch-size must divide the CIFAR-10 test set")
  if not (args.eval_batch_size * args.tta_fraction).is_integer():
    parser.error("--tta-fraction times --eval-batch-size must be an integer")
  batches_per_epoch = X_train_norm.shape[0] // batch_size
  epoch_count = math.ceil(steps / batches_per_epoch)
  whiten_bias_steps = math.ceil(0.2 * batches_per_epoch)
  wd = WEIGHT_DECAY * batch_size
  opt_whiten = nn.optim.SGD(whiten_bias, lr=BIAS_LR, momentum=SGD_MOMENTUM, nesterov=True,
                            weight_decay=wd/BIAS_LR, fused=False)
  opt_norm = nn.optim.SGD(norm_biases, lr=BIAS_LR, momentum=SGD_MOMENTUM, nesterov=True,
                          weight_decay=wd/BIAS_LR, fused=False)
  opt_head = nn.optim.SGD(head, lr=HEAD_LR, momentum=SGD_MOMENTUM, nesterov=True,
                          weight_decay=wd/HEAD_LR, fused=False)
  opt_muon = AirbenchMuon(hidden_convs, lr=MUON_LR, momentum=MUON_MOMENTUM, weight_decay=wd, ns_steps=3)
  phase("optim_init")

  mutable_tensors:list[Tensor] = []
  for tensor in [*state.values(), opt_whiten.lr, *opt_whiten.b, opt_norm.lr, *opt_norm.b,
                 opt_head.lr, *opt_head.b, opt_muon.lr, *opt_muon.b]:
    if all(tensor is not existing for existing in mutable_tensors): mutable_tensors.append(tensor)
  Tensor.realize(*mutable_tensors)
  initial_values = [tensor.clone().realize() for tensor in mutable_tensors]
  rng_values = {device: counter.clone().realize() for device, counter in Tensor._device_rng_counters.items()}
  permutation_rng = np.random.default_rng(args.seed)
  permutation_coefficients = []
  for _ in range(epoch_count):
    coefficients = np.empty((4, 4), dtype=np.int32)
    coefficients[:, :2] = permutation_rng.integers(0, batches_per_epoch, size=(4, 2), dtype=np.int32)
    coefficients[:, 2:] = permutation_rng.integers(0, batch_size, size=(4, 2), dtype=np.int32)
    permutation_coefficients.append(Tensor(coefficients, dtype=dtypes.int32).realize())

  def set_lrs(step:int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    whiten_lr = BIAS_LR * max(0.0, 1.0 - step / whiten_bias_steps)
    train_scale = max(0.0, 1.0 - step / steps)
    norm_lr = BIAS_LR * train_scale
    return Tensor([whiten_lr], dtype=dtypes.float32), Tensor([norm_lr], dtype=dtypes.float32), \
           Tensor([HEAD_LR * train_scale], dtype=dtypes.float32), Tensor([MUON_LR * train_scale], dtype=dtypes.float32)

  learning_rates = [set_lrs(step) for step in range(steps)]
  Tensor.realize(*[lr for step_lrs in learning_rates for lr in step_lrs])

  def zero_grads(*opts) -> None:
    for opt in opts: opt.zero_grad()

  def sgd_realize(lr_whiten:Tensor, lr_norm:Tensor, lr_head:Tensor) -> list[Tensor]:
    return [opt_whiten.lr.assign(lr_whiten), opt_norm.lr.assign(lr_norm), opt_head.lr.assign(lr_head),
            *opt_whiten.schedule_step(), *opt_norm.schedule_step(), *opt_head.schedule_step()]

  def muon_realize(lr_muon:Tensor, normalize_weights:bool) -> list[Tensor]:
    opt_muon.normalize_weights = normalize_weights
    return [opt_muon.lr.assign(lr_muon), *opt_muon.schedule_step()]

  def sgd_realize_no_whiten(lr_norm:Tensor, lr_head:Tensor) -> list[Tensor]:
    return [opt_norm.lr.assign(lr_norm), opt_head.lr.assign(lr_head), *opt_norm.schedule_step(), *opt_head.schedule_step()]

  def train_step_impl(X_epoch:Tensor, Y_epoch:Tensor, batch, lr_whiten:Tensor|None, lr_norm:Tensor,
                      lr_head:Tensor, lr_muon:Tensor, whiten_bias_grad:bool, normalize_weights:bool) -> Tensor:
    start = batch * batch_size
    X, Y = X_epoch[start:start+batch_size], Y_epoch[start:start+batch_size]
    zero_grads(opt_whiten, opt_norm, opt_head, opt_muon)
    loss = cross_entropy_sum(model(X, whiten_bias_grad=whiten_bias_grad), Y, LABEL_SMOOTHING)
    loss.backward()
    sgd = sgd_realize(lr_whiten, lr_norm, lr_head) if lr_whiten is not None else sgd_realize_no_whiten(lr_norm, lr_head)
    return loss.realize(*sgd, *muon_realize(lr_muon, normalize_weights), *bn_buffers)

  @TinyJit
  @Context(TRAINING=1)
  def train_step_bias(X:Tensor, Y:Tensor, batch, lr_whiten:Tensor, lr_norm:Tensor, lr_head:Tensor, lr_muon:Tensor) -> Tensor:
    return train_step_impl(X, Y, batch, lr_whiten, lr_norm, lr_head, lr_muon, True, False)

  @TinyJit
  @Context(TRAINING=1)
  def train_step_bias_norm(X:Tensor, Y:Tensor, batch, lr_whiten:Tensor, lr_norm:Tensor, lr_head:Tensor, lr_muon:Tensor) -> Tensor:
    return train_step_impl(X, Y, batch, lr_whiten, lr_norm, lr_head, lr_muon, True, True)

  @TinyJit
  @Context(TRAINING=1)
  def train_step(X:Tensor, Y:Tensor, batch, lr_norm:Tensor, lr_head:Tensor, lr_muon:Tensor) -> Tensor:
    return train_step_impl(X, Y, batch, None, lr_norm, lr_head, lr_muon, False, False)

  @TinyJit
  @Context(TRAINING=1)
  def train_step_norm(X:Tensor, Y:Tensor, batch, lr_norm:Tensor, lr_head:Tensor, lr_muon:Tensor) -> Tensor:
    return train_step_impl(X, Y, batch, None, lr_norm, lr_head, lr_muon, False, True)

  @TinyJit
  def prepare_epoch(Xsrc:Tensor, indices:Tensor) -> tuple[Tensor, Tensor]:
    X = random_crop_batch(Xsrc, indices)
    X = batch_color_jitter(X, BRIGHTNESS, CONTRAST)
    return activation_buffer(X).realize(), Y_train[indices].contiguous().realize()

  @TinyJit
  def prepare_permutation(coefficients:Tensor) -> Tensor:
    return random_permutation(batches_per_epoch, batch_size, coefficients).realize()

  def infer_mirror(X:Tensor) -> Tensor:
    return (model(X, whiten_bias_grad=False) + model(X.flip(-1), whiten_bias_grad=False)) * 0.5

  def infer_tta2(X:Tensor) -> Tensor:
    Xp = pad_reflect(X, 1)
    base_views = Tensor.cat(X, Xp[:, :, 0:32, 0:32], Xp[:, :, 2:34, 2:34], dim=0)
    logits = model(Tensor.cat(base_views, base_views.flip(-1), dim=0), whiten_bias_grad=False).reshape(6, X.shape[0], 10)
    base = (logits[0] + logits[3]) * 0.5
    translated = (logits[1] + logits[2] + logits[4] + logits[5]) * 0.25
    return base * TTA_BASE_WEIGHT + translated * (1.0 - TTA_BASE_WEIGHT)

  @TinyJit
  @Context(TRAINING=0)
  def eval_step_basic(i) -> Tensor:
    X, Y = X_test[i:i+args.eval_batch_size], Y_test[i:i+args.eval_batch_size]
    return (model(X, whiten_bias_grad=False).argmax(axis=1) == Y).sum().realize()

  @TinyJit
  @Context(TRAINING=0)
  def eval_step_mirror(i) -> Tensor:
    X, Y = X_test[i:i+args.eval_batch_size], Y_test[i:i+args.eval_batch_size]
    return (infer_mirror(X).argmax(axis=1) == Y).sum().realize()

  @TinyJit
  @Context(TRAINING=0)
  def eval_base_logits(i) -> Tensor:
    return model(X_test[i:i+args.eval_batch_size], whiten_bias_grad=False).realize()

  eval_i = Variable("eval_i", 0, X_test.shape[0] - args.eval_batch_size)
  selected_count = round(X_test.shape[0] * args.tta_fraction)
  selected_batch_size = round(args.eval_batch_size * args.tta_fraction)
  assert selected_count % selected_batch_size == 0
  selected_i = Variable("selected_i", 0, selected_count - selected_batch_size)

  @TinyJit
  @Context(TRAINING=0)
  def eval_select_global(*chunks:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    logits = Tensor.cat(*chunks, dim=0)
    uncertain = select_tta_indices(logits, selected_count)
    base_correct = logits.argmax(axis=1) == Y_test
    return uncertain.realize(), base_correct.sum().realize(), base_correct[uncertain].sum().realize()

  @TinyJit
  @Context(TRAINING=0)
  def eval_selected_tta(indices:Tensor, i, correct:Tensor) -> Tensor:
    batch_indices = indices[i:i+selected_batch_size]
    X, Y = X_test[batch_indices], Y_test[batch_indices]
    return correct.assign(correct + (infer_tta2(X).argmax(axis=1) == Y).sum()).realize()

  @TinyJit
  def eval_finalize(base_correct:Tensor, base_selected:Tensor, tta_correct:Tensor) -> Tensor:
    return (base_correct - base_selected + tta_correct).realize()

  def evaluate() -> Tensor:
    assert X_test.shape[0] % args.eval_batch_size == 0, "eval batch size must divide CIFAR-10 test size"
    if args.tta_level == 2:
      chunks = [eval_base_logits(eval_i.bind(i)).clone().realize() for i in range(0, X_test.shape[0], args.eval_batch_size)]
      uncertain, base_correct, base_selected = eval_select_global(*chunks)
      selected_correct = Tensor.zeros((), dtype=dtypes.int32).clone().realize()
      for i in range(0, selected_count, selected_batch_size):
        eval_selected_tta(uncertain, selected_i.bind(i), selected_correct)
      return eval_finalize(base_correct, base_selected, selected_correct)
    eval_fn = (eval_step_basic, eval_step_mirror)[args.tta_level]
    correct = Tensor.zeros((), dtype=dtypes.int32).realize()
    for i in range(0, X_test.shape[0], args.eval_batch_size):
      correct = (correct + eval_fn(eval_i.bind(i)).cast(dtypes.int32)).realize()
    return correct

  norm_steps:set[int] = set()
  last_norm_step = 0
  for current_step in range(1, steps + 1):
    norm_frequency = 2 + int(15 * current_step / steps)
    if current_step - last_norm_step >= norm_frequency:
      norm_steps.add(current_step)
      last_norm_step = current_step
  train_batch = Variable("train_batch", 0, batches_per_epoch - 1)

  compile_warmup = 0.0
  if not args.no_warmup:
    Device[Device.DEFAULT].synchronize()
    warmup_start = time.perf_counter()
    for _ in range(2): warm_indices = prepare_permutation(permutation_coefficients[0])
    for _ in range(2): X_warm, Y_warm = prepare_epoch(X_train_pad, warm_indices)
    warm_batch = train_batch.bind(0)
    warm_lrs = learning_rates[0]
    for _ in range(2): train_step_bias(X_warm, Y_warm, warm_batch, *warm_lrs)
    for _ in range(2): train_step_bias_norm(X_warm, Y_warm, warm_batch, *warm_lrs)
    for _ in range(2): train_step(X_warm, Y_warm, warm_batch, *warm_lrs[1:])
    for _ in range(2): train_step_norm(X_warm, Y_warm, warm_batch, *warm_lrs[1:])
    whitening_images = X_train_norm[:960].contiguous().realize()
    for _ in range(2): whitening_covariance(whitening_images)
    if args.tta_level == 2:
      warm_chunks = []
      for i in range(0, X_test.shape[0], args.eval_batch_size):
        warm_chunks.append(eval_base_logits(eval_i.bind(i)).clone().realize())
      for _ in range(2): warm_selected, warm_base, warm_base_selected = eval_select_global(*warm_chunks)
      warm_correct = Tensor.zeros((), dtype=dtypes.int32).clone().realize()
      for _ in range(2): eval_selected_tta(warm_selected, selected_i.bind(0), warm_correct)
      for _ in range(2): eval_finalize(warm_base, warm_base_selected, warm_correct)
    else:
      eval_fn = (eval_step_basic, eval_step_mirror)[args.tta_level]
      for _ in range(2): eval_fn(eval_i.bind(0))
    Device[Device.DEFAULT].synchronize()
    compile_warmup = time.perf_counter() - warmup_start
    Tensor.realize(*[tensor.assign(value) for tensor, value in zip(mutable_tensors, initial_values)])
    Tensor.realize(*[Tensor._device_rng_counters[device].assign(value) for device, value in rng_values.items()])
    Device[Device.DEFAULT].synchronize()
    phase("compile_warmup_restore")

  Device[Device.DEFAULT].synchronize()
  t0 = time.perf_counter()
  init_whitening_(model, X_train_norm, n=960)
  phase("whitening_init")

  if args.profile_phases: Device[Device.DEFAULT].synchronize()
  train_start = time.perf_counter()
  segment_start, segment_step = train_start, 0
  step = 0
  for epoch in range(epoch_count):
    if args.profile_phases:
      Device[Device.DEFAULT].synchronize()
      epoch_prepare_start = time.perf_counter()
    Xsrc = X_train_pad if epoch % 2 == 0 else X_train_pad_flip
    indices = prepare_permutation(permutation_coefficients[epoch])
    X_epoch, Y_epoch = prepare_epoch(Xsrc, indices)
    if args.profile_phases:
      Device[Device.DEFAULT].synchronize()
      print(f"phase=epoch_prepare_{epoch} seconds={time.perf_counter()-epoch_prepare_start:.4f}", flush=True)
    for epoch_step in range(batches_per_epoch):
      lr_whiten, lr_norm, lr_head, lr_muon = learning_rates[step]
      detail_profile = args.profile_phases and steps <= 10
      if detail_profile:
        Device[Device.DEFAULT].synchronize()
        detail_start = time.perf_counter()
      batch = train_batch.bind(epoch_step)
      normalize_weights = step + 1 in norm_steps
      if step < whiten_bias_steps:
        train_fn = train_step_bias_norm if normalize_weights else train_step_bias
        train_fn(X_epoch, Y_epoch, batch, lr_whiten, lr_norm, lr_head, lr_muon)
      else:
        train_fn = train_step_norm if normalize_weights else train_step
        train_fn(X_epoch, Y_epoch, batch, lr_norm, lr_head, lr_muon)
      if detail_profile:
        Device[Device.DEFAULT].synchronize()
        print(f"phase=train_graph_step_{step} seconds={time.perf_counter()-detail_start:.4f}", flush=True)
      step += 1
      if args.profile_phases and step in {2, whiten_bias_steps, whiten_bias_steps + 2, steps}:
        Device[Device.DEFAULT].synchronize()
        now = time.perf_counter()
        print(f"phase=train_steps_{segment_step}_{step} seconds={now-segment_start:.4f}", flush=True)
        segment_start, segment_step = now, step
      if step >= steps: break
  if args.profile_phases and not args.quiet:
    Device[Device.DEFAULT].synchronize()
    train_end = time.perf_counter()
    print(f"phase=timed_train seconds={train_end-train_start:.4f}", flush=True)
  eval_start = time.perf_counter()
  correct = evaluate()
  correct_count = int(correct.numpy().item())
  Device[Device.DEFAULT].synchronize()
  end_time = time.perf_counter()
  wall_time = end_time - t0
  total_time = end_time - total_start
  if args.profile_phases and not args.quiet:
    print(f"phase=timed_eval seconds={time.perf_counter()-eval_start:.4f}", flush=True)

  acc = correct_count / X_test.shape[0] * 100.0
  if not args.quiet:
    gpu_id = os.getenv("SLURM_JOB_GPUS", os.getenv("CUDA_VISIBLE_DEVICES", "")).split(",")[0]
    gpu_query = ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"]
    if gpu_id: gpu_query += ["-i", gpu_id]
    try: gpu_info = subprocess.check_output(gpu_query, text=True).strip().splitlines()[0]
    except Exception: gpu_info = "unknown"
    try:
      commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
      dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception: commit, dirty = "unknown", "unknown"
    print(f"hardware={gpu_info} tinygrad_commit={commit} git_dirty={dirty}")
    print(f"runtime DEV={Device.DEFAULT} DEFAULT_FLOAT={dtypes.default_float} JITBEAM={getenv('JITBEAM', 0)}")
  print(f"device={Device.DEFAULT} seed={args.seed} steps={steps} batch_size={batch_size} width={WIDTH} "
        f"whiten_bias_steps={whiten_bias_steps} tta_level={args.tta_level} tta_base_weight={TTA_BASE_WEIGHT:.2f} "
        f"tta_fraction={args.tta_fraction:.3f} tta_score=margin")
  print(f"optimizer bias_lr={BIAS_LR:.6f} head_lr={HEAD_LR:.6f} muon_lr={MUON_LR:.6f} "
        f"sgd_momentum={SGD_MOMENTUM:.4f} muon_momentum={MUON_MOMENTUM:.4f} weight_decay={wd:.8f}")
  print(f"loss label_smoothing={LABEL_SMOOTHING:.4f} loss_mult=1.0000 "
        f"bn_eps={BN_EPS:.2g} bn_momentum={BN_MOMENTUM:.4f} logit_div={LOGIT_DIV:.4f}")
  print("benchmark_region=whitening_train_eval_accuracy_readback compile_warmup=excluded")
  print(f"accuracy={acc:.2f} correct={correct_count}/{X_test.shape[0]} benchmark_wall_time_s={wall_time:.4f} "
        f"compile_warmup_s={compile_warmup:.4f} after_import_wall_time_s={total_time:.4f}")
  if args.target_acc and acc < args.target_acc: raise SystemExit(f"accuracy {acc:.2f} < target {args.target_acc:.2f}")
  if args.target_time and wall_time >= args.target_time: raise SystemExit(f"benchmark wall time {wall_time:.4f} >= target {args.target_time:.4f}")
  if args.save_weights: nn.state.safe_save(nn.state.get_state_dict(model), args.save_weights)

if __name__ == "__main__":
  if TRAINING: raise RuntimeError("run without TRAINING in the environment")
  main()
