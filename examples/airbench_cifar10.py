#!/usr/bin/env python3
"""Train CIFAR-10 with a tinygrad-tuned Airbench/Hiverge recipe.

First-call TinyJit capture is included in the pipeline timer. Use an external timer for whole-process wall time.
The recipe is derived from MIT-licensed Airbench and Hiverge code, then retuned for tinygrad's NV backend on one L40S:
https://github.com/KellerJordan/cifar10-airbench/tree/4c1b6d1e3889b037efadcfd5c0ea65b246592362
https://github.com/hiverge/cifar10-speedrun/tree/06c60727547042c847d919c5848e807e8119d582

Run on one NVIDIA GPU with: DEV=NV JITBEAM=2 python3 examples/airbench_cifar10.py
"""
import argparse, math, time
from tinygrad import Tensor, TinyJit, Device, dtypes, nn, Variable
from tinygrad.helpers import Context, getenv, TRAINING

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NS_COEFFS = (3.4576, -4.7391, 2.0843)
BN_EPS, BN_MOMENTUM = 1e-12, getenv("BN_MOMENTUM", 0.4434)
BIAS_LR, HEAD_LR, MUON_LR = getenv("BIAS_LR", 0.0573), getenv("HEAD_LR", 0.540), getenv("MUON_LR", 0.2425)
SGD_MOMENTUM, MUON_MOMENTUM = getenv("SGD_MOMENTUM", 0.825), getenv("MUON_MOMENTUM", 0.655)
WEIGHT_DECAY = getenv("WEIGHT_DECAY", 1.0418e-6)
LABEL_SMOOTHING = getenv("LABEL_SMOOTHING", 0.09)
BRIGHTNESS, CONTRAST = getenv("BRIGHTNESS", 0.1399), getenv("CONTRAST", 0.1308)
WHITEN_BIAS_SCALE, WHITEN_BIAS_OFFSET = getenv("WHITEN_BIAS_SCALE", 0.80), getenv("WHITEN_BIAS_OFFSET", 0.3375)

def whiten_activ(x:Tensor) -> Tensor:
  return x.gelu("none")

def block_activ(x:Tensor) -> Tensor:
  return x.silu()

def activation_buffer(x:Tensor) -> Tensor:
  return x.contiguous()

def airbench_conv2d(x:Tensor, weight:Tensor) -> Tensor:
  """3x3 padded convolution with GEMM-shaped gradients for NVIDIA tensor cores."""
  n, cin, h, w = x.shape
  cout = weight.shape[0]
  patches = x.pad((1, 1, 1, 1))._pool((3, 3)).permute(0, 2, 3, 1, 4, 5).reshape(n * h * w, cin * 9)
  patches = patches.contiguous().contiguous_backward()
  out = (patches @ weight.reshape(cout, cin * 9).T).contiguous().contiguous_backward()
  return out.reshape(n, h, w, cout).permute(0, 3, 1, 2)

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
  def __init__(self, bn_eps:float, bn_momentum:float, width:int=256):
    self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
    self.whiten.weight.is_param_(False)
    self.whiten.bias.is_param_(False)
    self.block1 = ConvGroup(24, 64, bn_eps, bn_momentum)
    self.block2 = ConvGroup(64, width, bn_eps, bn_momentum)
    self.block3 = ConvGroup(width, width, bn_eps, bn_momentum)
    self.head = nn.Linear(width, 10, bias=False)
    self.logit_div = float(width)
    self.head.weight.replace((self.head.weight / self.head.weight.float().std()).cast(self.head.weight.dtype).clone())

  def __call__(self, x:Tensor) -> Tensor:
    # Aligned 32x32 feature maps are substantially faster than the whitening layer's native 31x31 output on tinygrad.
    x = activation_buffer(whiten_activ(x.conv2d(self.whiten.weight.detach(), self.whiten.bias.detach())).pad((1, 0, 0, 1)))
    x = activation_buffer(self.block1(x))
    x = activation_buffer(self.block2(x))
    x = activation_buffer(self.block3(x))
    x = x.max_pool2d(3).reshape(x.shape[0], -1)
    return self.head(x) / self.logit_div

class AirbenchMuon(nn.optim.Optimizer):
  def __init__(self, params:list[Tensor], lr=0.205, momentum=0.655, weight_decay=0.0, ns_steps=3):
    super().__init__(params, lr, fused=False)
    self.momentum, self.weight_decay, self.ns_steps = momentum, weight_decay, ns_steps
    self.normalize_weights:bool|Tensor = False
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
      if self.normalize_weights is not False:
        normalized = base * ((p.shape[0] ** 0.5) / (base.float().square().sum().sqrt() + 1e-7))
        base = normalized if self.normalize_weights is True else self.normalize_weights.where(normalized, base)
      updated = (base - self.lr * g.cast(p.dtype)) * (1.0 - self.lr * self.weight_decay)
      updates.append((p.detach() - updated).cast(p.dtype))
    return updates, self.b

def dirac_init_(conv:nn.Conv2d) -> None:
  # Match Airbench: preserve random expansion channels, Dirac only the first input-width block.
  oc, ic, kh, kw = conv.weight.shape
  n = min(oc, ic)
  identity = Tensor.eye(n, dtype=conv.weight.dtype).reshape(n, n, 1, 1)
  identity = identity.pad((kw//2, kw-kw//2-1, kh//2, kh-kh//2-1, 0, ic-n))
  weight = identity.cat(conv.weight[n:].detach(), dim=0) if oc > n else identity
  conv.weight.replace(weight.clone())

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

def symmetric_eigh(matrix:list[list[float]], tolerance=1e-10) -> tuple[list[float], list[list[float]]]:
  """Eigenvalues and column eigenvectors of a small real symmetric matrix using Jacobi rotations."""
  size = len(matrix)
  if size == 0 or any(len(row) != size for row in matrix): raise ValueError("matrix must be non-empty and square")
  values = [list(map(float, row)) for row in matrix]
  vectors = [[float(row == col) for col in range(size)] for row in range(size)]
  for _ in range(100 * size * size):
    p, q = 0, 1
    largest = abs(values[p][q]) if size > 1 else 0.0
    for row in range(size - 1):
      for col in range(row + 1, size):
        if abs(values[row][col]) > largest: p, q, largest = row, col, abs(values[row][col])
    if largest <= tolerance: break

    app, aqq, apq = values[p][p], values[q][q], values[p][q]
    tau = (aqq - app) / (2.0 * apq)
    tangent = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
    cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
    sine = tangent * cosine
    for idx in range(size):
      if idx in (p, q): continue
      aip, aiq = values[idx][p], values[idx][q]
      values[idx][p] = values[p][idx] = cosine * aip - sine * aiq
      values[idx][q] = values[q][idx] = sine * aip + cosine * aiq
    values[p][p] = cosine*cosine*app - 2.0*sine*cosine*apq + sine*sine*aqq
    values[q][q] = sine*sine*app + 2.0*sine*cosine*apq + cosine*cosine*aqq
    values[p][q] = values[q][p] = 0.0
    for row in range(size):
      vip, viq = vectors[row][p], vectors[row][q]
      vectors[row][p] = cosine * vip - sine * viq
      vectors[row][q] = sine * vip + cosine * viq
  else: raise RuntimeError("symmetric eigendecomposition did not converge")

  order = sorted(range(size), key=lambda idx: values[idx][idx])
  eigenvalues = [values[idx][idx] for idx in order]
  eigenvectors = [[vectors[row][idx] for idx in order] for row in range(size)]
  for col in range(size):
    pivot = max(range(size), key=lambda row: abs(eigenvectors[row][col]))
    if eigenvectors[pivot][col] < 0.0:
      for row in range(size): eigenvectors[row][col] = -eigenvectors[row][col]
  return eigenvalues, eigenvectors

def permutation_parameters(seed:int, epochs:int, rows:int, cols:int) -> Tensor:
  """Generate deterministic modular-shuffle coefficients without importing a numerical RNG library."""
  state, mask = seed & 0xffffffffffffffff, 0xffffffffffffffff
  def randbelow(limit:int) -> int:
    nonlocal state
    state = (state + 0x9e3779b97f4a7c15) & mask
    value = state
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    value ^= value >> 31
    return (value * limit) >> 64
  return Tensor([[[randbelow(rows), randbelow(rows), randbelow(cols), randbelow(cols)] for _ in range(4)]
                 for _ in range(epochs)], dtype=dtypes.int32).realize()

@Context(ALLOW_TF32=0)
def whitening_covariance(images:Tensor) -> Tensor:
  patches = images.float()._pool((2, 2)).permute(0, 2, 3, 1, 4, 5).reshape(-1, 12)
  return ((patches.T @ patches) / patches.shape[0]).realize()

@Context(ALLOW_TF32=0)
def whitening_patch_mean(images:Tensor) -> Tensor:
  patches = images.float()._pool((2, 2)).permute(0, 2, 3, 1, 4, 5).reshape(-1, 12)
  return patches.mean(axis=0).realize()

def init_whitening_(model:AirbenchCifarNet, train_images:Tensor, eps=5e-4, n=5000) -> None:
  images = train_images[:n].contiguous().realize()
  vals, vecs = symmetric_eigh(whitening_covariance(images).tolist())
  w12 = [[vecs[row][col] / math.sqrt(vals[col] + eps) for row in range(12)] for col in range(12)]
  w = w12 + [[-value for value in row] for row in w12]
  model.whiten.weight.replace(Tensor(w, dtype=dtypes.float32).reshape(24, 3, 2, 2).cast(dtypes.default_float).clone().realize())
  patch_mean = whitening_patch_mean(images).tolist()
  bias = [-sum(value * mean for value, mean in zip(row, patch_mean)) * WHITEN_BIAS_SCALE + WHITEN_BIAS_OFFSET for row in w]
  model.whiten.bias.assign(Tensor(bias, dtype=model.whiten.bias.dtype)).realize()

def preprocess_cifar() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
  X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
  mean = Tensor(CIFAR_MEAN, dtype=dtypes.default_float).reshape(1,3,1,1)
  std = Tensor(CIFAR_STD, dtype=dtypes.default_float).reshape(1,3,1,1)
  scale = Tensor(255.0, dtype=dtypes.default_float)
  X_train = ((X_train.cast(dtypes.default_float) / scale - mean) / std).realize()
  X_test = activation_buffer((X_test.cast(dtypes.default_float) / scale - mean) / std).realize()
  X_train_pad = pad_reflect(batch_random_flip(X_train), 2).realize()
  X_train_pad_flip = X_train_pad.flip(-1).contiguous().realize()
  return X_train_pad, X_train_pad_flip, X_train, X_test.realize(), Y_train.realize(), Y_test.realize()

def cross_entropy_sum(logits:Tensor, labels:Tensor, label_smoothing:float) -> Tensor:
  return logits.float().sparse_categorical_crossentropy(labels, reduction="sum", label_smoothing=label_smoothing)

def main() -> None:
  parser = argparse.ArgumentParser(description="Airbench/Hiverge CIFAR-10 speed benchmark in tinygrad")
  parser.add_argument("--batch-size", type=int, default=getenv("BS", 2000))
  parser.add_argument("--steps", type=int, default=getenv("STEPS", 142))
  parser.add_argument("--width", type=int, default=getenv("WIDTH", 224))
  parser.add_argument("--muon-normalization", choices=("periodic", "always", "never"), default="periodic")
  parser.add_argument("--eval-batch-size", type=int, default=getenv("EVAL_BS", 2000))
  parser.add_argument("--tta-level", type=int, choices=(0, 1), default=getenv("TTA_LEVEL", 1),
                      help="0 disables TTA, 1 averages each image with its mirror")
  parser.add_argument("--tta-base-weight", type=float, default=getenv("TTA_BASE_WEIGHT", 0.505),
                      help="weight of the original image when mirror TTA is enabled")
  parser.add_argument("--seed", type=int, default=getenv("SEED", 6))
  parser.add_argument("--target-acc", type=float, default=getenv("TARGET_EVAL_ACC_PCT", 93.5))
  parser.add_argument("--target-time", type=float, default=getenv("TARGET_TIME_S", 10.0))
  parser.add_argument("--quiet", action="store_true", default=bool(getenv("QUIET", 0)))
  parser.add_argument("--profile-phases", action="store_true", default=bool(getenv("PROFILE_PHASES", 0)))
  parser.add_argument("--save-weights", type=str, default="")
  args = parser.parse_args()
  if args.batch_size <= 0: parser.error("--batch-size must be positive")
  if args.steps <= 0: parser.error("--steps must be positive")
  if args.width <= 0: parser.error("--width must be positive")
  if args.eval_batch_size <= 0: parser.error("--eval-batch-size must be positive")
  if not 0.0 <= args.tta_base_weight <= 1.0: parser.error("--tta-base-weight must be between 0 and 1")

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
  model = AirbenchCifarNet(BN_EPS, BN_MOMENTUM, args.width)
  phase("model_init")
  X_train_pad, X_train_pad_flip, X_train_norm, X_test, Y_train, Y_test = preprocess_cifar()
  phase("data_preprocess")

  state = nn.state.get_state_dict(model)
  groups = (model.block1, model.block2, model.block3)
  hidden_convs = [conv.conv.weight for group in groups for conv in (group.conv1, group.conv2)]
  norm_biases = [norm.bias for group in groups for norm in (group.norm1, group.norm2)]
  head = [model.head.weight]
  bn_buffers = [v for k,v in state.items() if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k]

  batch_size, steps = args.batch_size, args.steps
  if X_train_norm.shape[0] % batch_size: parser.error("--batch-size must divide the CIFAR-10 training set")
  if X_test.shape[0] % args.eval_batch_size: parser.error("--eval-batch-size must divide the CIFAR-10 test set")
  batches_per_epoch = X_train_norm.shape[0] // batch_size
  epoch_count = math.ceil(steps / batches_per_epoch)
  wd = WEIGHT_DECAY * batch_size
  opt_norm = nn.optim.SGD(norm_biases, lr=BIAS_LR, momentum=SGD_MOMENTUM, nesterov=True,
                          weight_decay=wd/BIAS_LR, fused=False)
  opt_head = nn.optim.SGD(head, lr=HEAD_LR, momentum=SGD_MOMENTUM, nesterov=True,
                          weight_decay=wd/HEAD_LR, fused=False)
  opt_muon = AirbenchMuon(hidden_convs, lr=MUON_LR, momentum=MUON_MOMENTUM, weight_decay=wd, ns_steps=3)
  phase("optim_init")

  Tensor.realize(*state.values(), opt_norm.lr, *opt_norm.b, opt_head.lr, *opt_head.b, opt_muon.lr, *opt_muon.b)
  permutation_coefficients = permutation_parameters(args.seed, epoch_count, batches_per_epoch, batch_size)

  def set_lrs(step:int) -> tuple[float, float, float]:
    train_scale = max(0.0, 1.0 - step / steps)
    norm_lr = BIAS_LR * train_scale
    return norm_lr, HEAD_LR * train_scale, MUON_LR * train_scale

  learning_rates = Tensor([set_lrs(step) for step in range(steps)], dtype=dtypes.float32).realize()
  train_step_number = Variable("train_step", 0, steps - 1)

  def zero_grads(*opts) -> None:
    for opt in opts: opt.zero_grad()

  def sgd_realize(lr_norm:Tensor, lr_head:Tensor) -> list[Tensor]:
    return [opt_norm.lr.assign(lr_norm), opt_head.lr.assign(lr_head), *opt_norm.schedule_step(), *opt_head.schedule_step()]

  def muon_realize(lr_muon:Tensor, normalize_weights:Tensor) -> list[Tensor]:
    opt_muon.normalize_weights = normalize_weights
    return [opt_muon.lr.assign(lr_muon), *opt_muon.schedule_step()]

  def train_step_impl(X_epoch:Tensor, Y_epoch:Tensor, batch, step) -> Tensor:
    start = batch * batch_size
    X, Y = X_epoch[start:start+batch_size], Y_epoch[start:start+batch_size]
    lr_norm, lr_head, lr_muon = [learning_rates[step, i:i+1] for i in range(3)]
    normalize_weights = normalization_schedule[step]
    zero_grads(opt_norm, opt_head, opt_muon)
    loss = cross_entropy_sum(model(X), Y, LABEL_SMOOTHING)
    loss.backward()
    return loss.realize(*sgd_realize(lr_norm, lr_head), *muon_realize(lr_muon, normalize_weights), *bn_buffers)

  @Context(TRAINING=1)
  def train_step_fxn(X:Tensor, Y:Tensor, batch, step) -> Tensor:
    return train_step_impl(X, Y, batch, step)

  train_step = TinyJit(train_step_fxn, warmup=False)

  def prepare_epoch(Xsrc:Tensor, indices:Tensor) -> tuple[Tensor, Tensor]:
    X = random_crop_batch(Xsrc, indices)
    X = batch_color_jitter(X, BRIGHTNESS, CONTRAST)
    return activation_buffer(X).realize(), Y_train[indices].contiguous().realize()

  def prepare_permutation(coefficients:Tensor, epoch) -> Tensor:
    return random_permutation(batches_per_epoch, batch_size, coefficients[epoch]).realize()

  prepare_epoch, prepare_permutation = TinyJit(prepare_epoch, warmup=False), TinyJit(prepare_permutation, warmup=False)
  tta_base_weight = Tensor([args.tta_base_weight], dtype=dtypes.default_float).realize()

  def infer_mirror(X:Tensor) -> Tensor:
    views = Tensor.cat(X, X.flip(-1), dim=0)
    logits = model(views).reshape(2, X.shape[0], 10)
    return logits[0] * tta_base_weight + logits[1] * (1.0 - tta_base_weight)

  @Context(TRAINING=0)
  def eval_step_basic(i) -> Tensor:
    X, Y = X_test[i:i+args.eval_batch_size], Y_test[i:i+args.eval_batch_size]
    return (model(X).argmax(axis=1) == Y).sum().realize()

  @Context(TRAINING=0)
  def eval_step_mirror(i) -> Tensor:
    X, Y = X_test[i:i+args.eval_batch_size], Y_test[i:i+args.eval_batch_size]
    return (infer_mirror(X).argmax(axis=1) == Y).sum().realize()

  eval_i = Variable("eval_i", 0, X_test.shape[0] - args.eval_batch_size)
  eval_step_basic, eval_step_mirror = TinyJit(eval_step_basic, warmup=False), TinyJit(eval_step_mirror, warmup=False)

  def evaluate() -> Tensor:
    assert X_test.shape[0] % args.eval_batch_size == 0, "eval batch size must divide CIFAR-10 test size"
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
  normalization_schedule = Tensor([
    args.muon_normalization == "always" or (args.muon_normalization == "periodic" and step + 1 in norm_steps)
    for step in range(steps)], dtype=dtypes.bool).realize()
  train_batch = Variable("train_batch", 0, batches_per_epoch - 1)
  train_epoch = Variable("train_epoch", 0, epoch_count - 1)

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
    indices = prepare_permutation(permutation_coefficients, train_epoch.bind(epoch))
    X_epoch, Y_epoch = prepare_epoch(Xsrc, indices)
    if args.profile_phases:
      Device[Device.DEFAULT].synchronize()
      print(f"phase=epoch_prepare_{epoch} seconds={time.perf_counter()-epoch_prepare_start:.4f}", flush=True)
    for epoch_step in range(batches_per_epoch):
      detail_profile = args.profile_phases and steps <= 10
      if detail_profile:
        Device[Device.DEFAULT].synchronize()
        detail_start = time.perf_counter()
      batch = train_batch.bind(epoch_step)
      step_var = train_step_number.bind(step)
      train_step(X_epoch, Y_epoch, batch, step_var)
      if detail_profile:
        Device[Device.DEFAULT].synchronize()
        print(f"phase=train_graph_step_{step} seconds={time.perf_counter()-detail_start:.4f}", flush=True)
      step += 1
      if args.profile_phases and step in {2, steps}:
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
  correct_count = int(correct.item())
  end_time = time.perf_counter()
  wall_time = end_time - t0
  total_time = end_time - total_start
  if args.profile_phases and not args.quiet:
    print(f"phase=timed_eval seconds={time.perf_counter()-eval_start:.4f}", flush=True)

  acc = correct_count / X_test.shape[0] * 100.0
  if not args.quiet:
    import os, subprocess
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
  print("jit first_call_capture=included train_graphs=1")
  print(f"device={Device.DEFAULT} seed={args.seed} steps={steps} batch_size={batch_size} width={args.width} "
        f"tta_level={args.tta_level} tta_base_weight={args.tta_base_weight:.3f}")
  print(f"muon_normalization={args.muon_normalization} whitening_bias=analytic_frozen")
  print(f"optimizer bias_lr={BIAS_LR:.6f} head_lr={HEAD_LR:.6f} muon_lr={MUON_LR:.6f} "
        f"sgd_momentum={SGD_MOMENTUM:.4f} muon_momentum={MUON_MOMENTUM:.4f} weight_decay={wd:.8f}")
  print(f"loss label_smoothing={LABEL_SMOOTHING:.4f} loss_mult=1.0000 "
        f"bn_eps={BN_EPS:.2g} bn_momentum={BN_MOMENTUM:.4f} logit_div={model.logit_div:.4f}")
  print("benchmark_region=whitening_train_eval_accuracy_readback first_call_jit_capture=included")
  print(f"accuracy={acc:.2f} correct={correct_count}/{X_test.shape[0]} benchmark_wall_time_s={wall_time:.4f} "
        f"after_import_wall_time_s={total_time:.4f}")
  if args.target_acc and acc < args.target_acc: raise SystemExit(f"accuracy {acc:.2f} < target {args.target_acc:.2f}")
  if args.target_time and wall_time >= args.target_time: raise SystemExit(f"benchmark wall time {wall_time:.4f} >= target {args.target_time:.4f}")
  if args.save_weights: nn.state.safe_save(nn.state.get_state_dict(model), args.save_weights)

if __name__ == "__main__":
  if TRAINING: raise RuntimeError("run without TRAINING in the environment")
  main()
