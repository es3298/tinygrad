#!/usr/bin/env python3
import argparse, math, subprocess, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes, nn, Variable
from tinygrad.helpers import Context, getenv, TRAINING

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NS_COEFFS = (3.4445, -4.7750, 2.0315)
EXACT_GELU = getenv("EXACT_GELU", 1)

def activ(x:Tensor) -> Tensor:
  return x.gelu("none") if EXACT_GELU else x.quick_gelu()

class AirbenchBatchNorm(nn.BatchNorm2d):
  def __init__(self, channels:int, eps:float):
    super().__init__(channels, eps=eps, momentum=0.4, affine=True, track_running_stats=True)
    self.weight = Tensor.ones(channels, dtype=dtypes.float32).is_param_(False)
    self.bias = Tensor.zeros(channels, dtype=dtypes.float32)
    self.running_mean = Tensor.zeros(channels, dtype=dtypes.float32).is_param_(False)
    self.running_var = Tensor.ones(channels, dtype=dtypes.float32).is_param_(False)
    self.weight.is_param_(False)

class Conv:
  def __init__(self, channels_in:int, channels_out:int):
    self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False)
    dirac_init_(self.conv)

  def __call__(self, x:Tensor) -> Tensor:
    return self.conv(x)

class ConvGroup:
  def __init__(self, channels_in:int, channels_out:int, bn_eps:float):
    self.conv1, self.conv2 = Conv(channels_in, channels_out), Conv(channels_out, channels_out)
    self.norm1, self.norm2 = AirbenchBatchNorm(channels_out, bn_eps), AirbenchBatchNorm(channels_out, bn_eps)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.conv1(x).max_pool2d(2).float()
    x = activ(self.norm1(x).cast(dtypes.default_float))
    x = self.conv2(x).float()
    return activ(self.norm2(x).cast(dtypes.default_float))

class AirbenchCifarNet:
  def __init__(self, bn_eps:float, logit_div:float):
    self.whiten = nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True)
    self.whiten.weight.is_param_(False)
    self.block1 = ConvGroup(24, 64, bn_eps)
    self.block2 = ConvGroup(64, 256, bn_eps)
    self.block3 = ConvGroup(256, 256, bn_eps)
    self.head = nn.Linear(256, 10, bias=False)
    self.logit_div = logit_div
    self.head.weight.assign((self.head.weight / self.head.weight.float().std()).cast(self.head.weight.dtype))

  def __call__(self, x:Tensor, whiten_bias_grad=True) -> Tensor:
    bias = self.whiten.bias if whiten_bias_grad else self.whiten.bias.detach()
    x = activ(x.conv2d(self.whiten.weight, bias))
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x.max_pool2d(3).reshape(x.shape[0], -1)
    return self.head(x) / self.logit_div

class AirbenchMuon(nn.optim.Optimizer):
  def __init__(self, params:list[Tensor], lr=0.24, momentum=0.6, nesterov=True, ns_steps=3, use_bf16=True):
    super().__init__(params, lr, fused=False)
    self.momentum, self.nesterov, self.ns_steps, self.use_bf16 = momentum, nesterov, ns_steps, use_bf16
    self.b = self._new_optim_param()

  def _step(self, params:list[Tensor], grads:list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    updates = []
    for i, (p, g) in enumerate(zip(params, grads)):
      self.b[i].assign(self.momentum * self.b[i] + g)
      g = (g + self.momentum * self.b[i]) if self.nesterov else self.b[i]
      g = g.reshape(g.shape[0], -1).cast(dtypes.bfloat16 if self.use_bf16 else dtypes.half).newton_schulz(self.ns_steps, NS_COEFFS).reshape(g.shape)
      p_norm = p.detach() * ((p.shape[0] ** 0.5) / (p.detach().float().square().sum().sqrt() + 1e-12))
      updates.append((p.detach() - p_norm + self.lr * g.cast(p.dtype)).cast(p.dtype))
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

def random_crop(X:Tensor, crop_size=32) -> Tensor:
  BS, _, H, W = X.shape
  low_x = Tensor.randint(BS, low=0, high=W-crop_size+1).reshape(BS,1,1,1)
  low_y = Tensor.randint(BS, low=0, high=H-crop_size+1).reshape(BS,1,1,1)
  idx_x = Tensor.arange(crop_size, dtype=dtypes.int32).reshape((1,1,1,crop_size))
  idx_y = Tensor.arange(crop_size, dtype=dtypes.int32).reshape((1,1,crop_size,1))
  return X.gather(-1, (low_x + idx_x).expand(-1, 3, X.shape[2], -1)).gather(-2, (low_y + idx_y).expand(-1, 3, crop_size, crop_size))

def batch_random_flip(X:Tensor) -> Tensor:
  return (Tensor.rand(X.shape[0], 1, 1, 1) < 0.5).where(X.flip(-1), X)

def init_whitening_(model:AirbenchCifarNet, train_images:Tensor, eps=5e-4, n=5000) -> None:
  patches = np.lib.stride_tricks.sliding_window_view(train_images[:n].float().numpy(), window_shape=(2,2), axis=(2,3))
  patches = patches.transpose((0,3,2,1,4,5)).reshape((-1, 3, 2, 2))
  flat = patches.reshape(len(patches), -1)
  cov = (flat.T @ flat) / len(flat)
  vals, vecs = np.linalg.eigh(cov, UPLO="U")
  w12 = (vecs.T.reshape(-1, 3, 2, 2) / np.sqrt(vals.reshape(-1,1,1,1) + eps)).astype(np.float32)
  w = np.concatenate((w12, -w12), axis=0)
  model.whiten.weight.assign(Tensor(w, dtype=dtypes.float32).cast(dtypes.default_float)).realize()

def preprocess_cifar() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
  X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
  mean = Tensor(CIFAR_MEAN, dtype=dtypes.float32).reshape(1,3,1,1)
  std = Tensor(CIFAR_STD, dtype=dtypes.float32).reshape(1,3,1,1)
  X_train = ((X_train.float() / 255.0 - mean) / std).cast(dtypes.default_float).realize()
  X_test = ((X_test.float() / 255.0 - mean) / std).cast(dtypes.default_float).realize()
  X_flip = batch_random_flip(X_train).realize()
  return pad_reflect(X_flip, 2).realize(), pad_reflect(X_flip.flip(-1), 2).realize(), X_train, X_test.realize(), Y_train.realize(), Y_test.realize()

def cross_entropy_sum(logits:Tensor, labels:Tensor) -> Tensor:
  return logits.float().sparse_categorical_crossentropy(labels, reduction="sum", label_smoothing=0.2)

def main() -> None:
  parser = argparse.ArgumentParser(description="Airbench/Muon CIFAR-10 speed benchmark in tinygrad")
  parser.add_argument("--batch-size", type=int, default=getenv("BS", 2000))
  parser.add_argument("--steps", type=int, default=getenv("STEPS", 200))
  parser.add_argument("--whiten-bias-steps", type=int, default=getenv("WHITEN_BIAS_STEPS", 75))
  parser.add_argument("--eval-batch-size", type=int, default=getenv("EVAL_BS", 2000))
  parser.add_argument("--tta-level", type=int, choices=(0, 1, 2), default=getenv("TTA_LEVEL", 2))
  parser.add_argument("--seed", type=int, default=getenv("SEED", 1337))
  parser.add_argument("--target-acc", type=float, default=getenv("TARGET_EVAL_ACC_PCT", 94.0))
  parser.add_argument("--target-time", type=float, default=getenv("TARGET_TIME_S", 10.0))
  parser.add_argument("--bias-lr", type=float, default=getenv("BIAS_LR", 0.053))
  parser.add_argument("--head-lr", type=float, default=getenv("HEAD_LR", 0.67))
  parser.add_argument("--muon-lr", type=float, default=getenv("MUON_LR", 0.24))
  parser.add_argument("--weight-decay", type=float, default=getenv("WEIGHT_DECAY", -1.0))
  parser.add_argument("--loss-mult", type=float, default=getenv("LOSS_MULT", -1.0))
  parser.add_argument("--bn-eps", type=float, default=getenv("BN_EPS", 1e-12))
  parser.add_argument("--logit-div", type=float, default=getenv("LOGIT_DIV", 256.0))
  parser.add_argument("--muon-fp16", action="store_true", default=bool(getenv("MUON_FP16", 0)))
  parser.add_argument("--no-muon", action="store_true", default=bool(getenv("NO_MUON", 0)))
  parser.add_argument("--no-sgd", action="store_true", default=bool(getenv("NO_SGD", 0)))
  parser.add_argument("--quiet", action="store_true", default=bool(getenv("QUIET", 0)))
  parser.add_argument("--profile-phases", action="store_true", default=bool(getenv("PROFILE_PHASES", 0)))
  parser.add_argument("--debug-stats", action="store_true", default=bool(getenv("DEBUG_STATS", 0)))
  parser.add_argument("--save-weights", type=str, default="")
  args = parser.parse_args()

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
  model = AirbenchCifarNet(args.bn_eps, args.logit_div)
  phase("model_init")
  X_train_pad, X_train_pad_flip, X_train_norm, X_test, Y_train, Y_test = preprocess_cifar()
  phase("data_preprocess")

  state = nn.state.get_state_dict(model)
  hidden_convs = [v for k,v in state.items() if v.is_param and v.ndim == 4 and not k.startswith("whiten.")]
  norm_biases = [v for k,v in state.items() if v.is_param and "norm" in k and k.endswith("bias")]
  whiten_bias = [model.whiten.bias]
  head = [model.head.weight]
  bn_buffers = [v for k,v in state.items() if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k]

  batch_size, steps = args.batch_size, args.steps
  loss_mult = 1.0 if args.loss_mult < 0 else args.loss_mult
  batches_per_epoch = X_train_norm.shape[0] // batch_size
  wd = 2e-6 * batch_size if args.weight_decay < 0 else args.weight_decay
  opt_whiten = nn.optim.SGD(whiten_bias, lr=args.bias_lr, momentum=0.85, nesterov=True, weight_decay=wd/args.bias_lr if args.bias_lr else 0.0, fused=False)
  opt_norm = nn.optim.SGD(norm_biases, lr=args.bias_lr, momentum=0.85, nesterov=True, weight_decay=wd/args.bias_lr if args.bias_lr else 0.0, fused=False)
  opt_head = nn.optim.SGD(head, lr=args.head_lr, momentum=0.85, nesterov=True, weight_decay=wd/args.head_lr if args.head_lr else 0.0, fused=False)
  opt_muon = AirbenchMuon(hidden_convs, lr=args.muon_lr, momentum=0.6, nesterov=True, ns_steps=3, use_bf16=not args.muon_fp16)
  phase("optim_init")

  Device[Device.DEFAULT].synchronize()
  t0 = time.perf_counter()
  init_whitening_(model, X_train_norm)
  phase("whitening_init")

  def set_lrs(step:int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    whiten_lr = args.bias_lr * max(0.0, 1.0 - step / args.whiten_bias_steps)
    train_scale = max(0.0, 1.0 - step / steps)
    norm_lr = args.bias_lr * train_scale
    return Tensor([whiten_lr], dtype=dtypes.float32), Tensor([norm_lr], dtype=dtypes.float32), Tensor([args.head_lr * train_scale], dtype=dtypes.float32), Tensor([args.muon_lr * train_scale], dtype=dtypes.float32)

  def zero_grads(*opts) -> None:
    for opt in opts: opt.zero_grad()

  def sgd_realize(lr_whiten:Tensor, lr_norm:Tensor, lr_head:Tensor) -> list[Tensor]:
    if args.no_sgd: return []
    return [opt_whiten.lr.assign(lr_whiten), opt_norm.lr.assign(lr_norm), opt_head.lr.assign(lr_head),
            *opt_whiten.schedule_step(), *opt_norm.schedule_step(), *opt_head.schedule_step()]

  def sgd_realize_no_whiten(lr_norm:Tensor, lr_head:Tensor) -> list[Tensor]:
    if args.no_sgd: return []
    return [opt_norm.lr.assign(lr_norm), opt_head.lr.assign(lr_head), *opt_norm.schedule_step(), *opt_head.schedule_step()]

  def muon_realize(lr_muon:Tensor) -> list[Tensor]:
    if args.no_muon: return []
    return [opt_muon.lr.assign(lr_muon), *opt_muon.schedule_step()]

  @TinyJit
  @Context(TRAINING=1)
  def train_step_bias(Xsrc:Tensor, idxs:Tensor, lr_whiten:Tensor, lr_norm:Tensor, lr_head:Tensor, lr_muon:Tensor) -> Tensor:
    X, Y = random_crop(Xsrc[idxs]), Y_train[idxs]
    zero_grads(opt_whiten, opt_norm, opt_head, opt_muon)
    loss = cross_entropy_sum(model(X, whiten_bias_grad=True), Y) * loss_mult
    loss.backward()
    return loss.realize(*sgd_realize(lr_whiten, lr_norm, lr_head), *muon_realize(lr_muon), *bn_buffers)

  @TinyJit
  @Context(TRAINING=1)
  def train_step(Xsrc:Tensor, idxs:Tensor, lr_norm:Tensor, lr_head:Tensor, lr_muon:Tensor) -> Tensor:
    X, Y = random_crop(Xsrc[idxs]), Y_train[idxs]
    zero_grads(opt_whiten, opt_norm, opt_head, opt_muon)
    loss = cross_entropy_sum(model(X, whiten_bias_grad=False), Y) * loss_mult
    loss.backward()
    return loss.realize(*sgd_realize_no_whiten(lr_norm, lr_head), *muon_realize(lr_muon), *bn_buffers)

  def infer_mirror(X:Tensor) -> Tensor:
    return (model(X, whiten_bias_grad=False) + model(X.flip(-1), whiten_bias_grad=False)) * 0.5

  def infer_tta2(X:Tensor) -> Tensor:
    logits = infer_mirror(X)
    Xp = pad_reflect(X, 1)
    logits_translate = (infer_mirror(Xp[:, :, 0:32, 0:32]) + infer_mirror(Xp[:, :, 2:34, 2:34])) * 0.5
    return logits * 0.5 + logits_translate * 0.5

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
  def eval_step_tta2(i) -> Tensor:
    X, Y = X_test[i:i+args.eval_batch_size], Y_test[i:i+args.eval_batch_size]
    return (infer_tta2(X).argmax(axis=1) == Y).sum().realize()

  def evaluate() -> Tensor:
    assert X_test.shape[0] % args.eval_batch_size == 0, "eval batch size must divide CIFAR-10 test size"
    eval_fn = (eval_step_basic, eval_step_mirror, eval_step_tta2)[args.tta_level]
    eval_i = Variable("eval_i", 0, X_test.shape[0] - args.eval_batch_size)
    correct = Tensor.zeros((), dtype=dtypes.int32).realize()
    for i in range(0, X_test.shape[0], args.eval_batch_size):
      correct = (correct + eval_fn(eval_i.bind(i)).cast(dtypes.int32)).realize()
    return correct

  Device[Device.DEFAULT].synchronize()
  train_start = time.perf_counter()
  step = 0
  for epoch in range(math.ceil(steps / batches_per_epoch)):
    Xsrc = X_train_pad if epoch % 2 == 0 else X_train_pad_flip
    indices = Tensor.randperm(X_train_norm.shape[0], dtype=dtypes.int32)[:batches_per_epoch*batch_size].reshape(batches_per_epoch, batch_size).realize()
    for epoch_step in range(batches_per_epoch):
      lr_whiten, lr_norm, lr_head, lr_muon = set_lrs(step)
      batch_idxs = indices[epoch_step].contiguous().realize()
      if step < args.whiten_bias_steps: train_step_bias(Xsrc, batch_idxs, lr_whiten, lr_norm, lr_head, lr_muon)
      else: train_step(Xsrc, batch_idxs, lr_norm, lr_head, lr_muon)
      step += 1
      if step >= steps: break
  if args.profile_phases and not args.quiet:
    Device[Device.DEFAULT].synchronize()
    train_end = time.perf_counter()
    print(f"phase=timed_train seconds={train_end-train_start:.4f}", flush=True)
  eval_start = time.perf_counter()
  correct = evaluate()
  Device[Device.DEFAULT].synchronize()
  end_time = time.perf_counter()
  wall_time = end_time - t0
  total_time = end_time - total_start
  if args.profile_phases and not args.quiet:
    print(f"phase=timed_eval seconds={time.perf_counter()-eval_start:.4f}", flush=True)

  correct_count = int(correct.numpy().item())
  acc = correct_count / X_test.shape[0] * 100.0
  if args.debug_stats and not args.quiet:
    logits = model(X_test[:args.eval_batch_size], whiten_bias_grad=False).realize()
    print(f"debug_logits min={float(logits.min().numpy().item()):.4f} max={float(logits.max().numpy().item()):.4f} mean={float(logits.mean().numpy().item()):.4f}", flush=True)
    param_nans, grad_nans, grad_seen = 0, 0, 0
    for p in nn.state.get_state_dict(model).values():
      param_nans += int(p.isnan().sum().numpy().item())
      if p.grad is not None:
        grad_seen += 1
        grad_nans += int(p.grad.isnan().sum().numpy().item())
    print(f"debug_nans params={param_nans} grads={grad_nans} grad_tensors={grad_seen}", flush=True)
  if args.save_weights:
    nn.state.safe_save(nn.state.get_state_dict(model), args.save_weights)
  if not args.quiet:
    try: gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except Exception: gpu_info = "unknown"
    try: commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: commit = "unknown"
    print(f"hardware={gpu_info} tinygrad_commit={commit}")
    print(f"runtime DEV={Device.DEFAULT} DEFAULT_FLOAT={dtypes.default_float}")
  print(f"device={Device.DEFAULT} seed={args.seed} steps={steps} batch_size={batch_size} tta_level={args.tta_level}")
  print("timed_region=whitening_train_eval")
  print(f"accuracy={acc:.2f} correct={correct_count}/{X_test.shape[0]} wall_time_s={wall_time:.4f} end_to_end_after_download_s={total_time:.4f}")
  if args.target_acc and acc < args.target_acc: raise SystemExit(f"accuracy {acc:.2f} < target {args.target_acc:.2f}")
  if args.target_time and wall_time > args.target_time: raise SystemExit(f"wall_time {wall_time:.4f} > target {args.target_time:.4f}")

if __name__ == "__main__":
  if TRAINING: raise RuntimeError("run without TRAINING in the environment")
  main()
