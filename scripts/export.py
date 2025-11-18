import logging
import pdb
import math
import os
import sys

logging.basicConfig(level=logging.INFO)
logging.info("library loading")
logging.info("DEBUG")
import torch


torch.set_grad_enabled(False)

import cached_conv as cc
import gin
import nn_tilde
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from absl import flags, app
from typing import Union, Optional
from torch.jit import script


try:
    import rave
except:
    import sys, os

    sys.path.append(os.path.abspath("."))
    import rave
import rave.blocks
import rave.core
import rave.resampler
from rave.prior import model as prior

# haptic modifications
from rave.pqmf import PQMF

FLAGS = flags.FLAGS


flags.DEFINE_string("run", default=None, help="Path to the run to export", required=True)
flags.DEFINE_bool("streaming", default=False, help="Enable the model streaming mode")
flags.DEFINE_float(
    "fidelity",
    default=0.95,
    lower_bound=0.1,
    upper_bound=0.999,
    help="Fidelity to use during inference (Variational mode only)",
)

flags.DEFINE_string("name", default=None, help="custom name for the scripted model (default: run name)")
flags.DEFINE_string("output", default=None, help="output location of scripted model")
flags.DEFINE_bool("ema_weights", default=False, help="Use ema weights if avaiable")
flags.DEFINE_integer("channels", default=None, help="number of out channels for export")
flags.DEFINE_integer("sr", default=None, help="Optional resampling sample rate")
flags.DEFINE_string("prior", default=None, help="path to prior (optional)")


class DumbPrior(nn.Module):

    def __init__(self, latent_size: int):
        super().__init__()
        self.ratio = 1
        self.latent_size = latent_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.dim() != 3:

        batch_size, latent_size, time_steps = x.size()
        return x.new_zeros(batch_size, self.latent_size, time_steps)


class HapticDecoderWrapper(nn.Module):
    """
    Wraps GeneratorV2 to safely return only the haptic prediction for JIT/nn_tilde.
    Handles cases where the generator returns (audio, haptic) or just (haptic).
    """

    def __init__(self, generator_v2_decoder: nn.Module):
        super().__init__()
        self.generator = generator_v2_decoder

    def forward(self, z):

        output = self.generator(z)

        if isinstance(output, tuple):

            haptic_pred = output[1]
        else:

            haptic_pred = output

        return haptic_pred


class IdentityModule(nn.Module):
    """A dummy module that returns its input, used to pass a JIT-compatible object when a module is None."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    # --- FIX: Add the expected Resampler methods ---
    @torch.jit.export
    def to_model_sampling_rate(self, x: torch.Tensor) -> torch.Tensor:
        # Calls the standard forward method, which simply returns x
        return self.forward(x)

    @torch.jit.export
    def from_model_sampling_rate(self, x: torch.Tensor) -> torch.Tensor:
        # Calls the standard forward method, which simply returns x
        return self.forward(x)

    # -----------------------------------------------


from torch.jit import ScriptModule


# @torch.jit.ignore  # Exclude this function from TorchScript
# def _process_output(
#     y: torch.Tensor,
#     n_batch: torch.Tensor,
#     target_channels: int,  # Keep target_channels as int
#     n_channels: int,  # Keep n_channels as int
#     stereo_mode: bool,  # Keep stereo_mode as bool
#     # resampler: Optional[nn.Module],
#     # resampler: Optional[torch.jit.ScriptModule],
#     resampler,  # OTHER OPTION than ABOVE
#     # resampler: Union[nn.Module, None],
# ) -> torch.Tensor:
#     """
#     Processes the output tensor `y` based on target channels, batch size, stereo mode, and resampling.
#     """
#     n_batch_int = int(n_batch.item())  # Convert Tensor to Python int

#     if stereo_mode:  # stereo_mode is now a Python bool
#         n_batch_int = int(n_batch_int / 2)
#         y = torch.cat([y[:n_batch_int], y[n_batch_int:]], dim=1)
#     elif target_channels > n_channels:  # target_channels/n_channels are Python ints
#         y = torch.cat(y.chunk(target_channels, 0), dim=1)
#     elif target_channels < n_channels:
#         y = y[:, :target_channels]

#     # if resampler is not None:
#     y = resampler.from_model_sampling_rate(y)

#     return y


@torch.jit.ignore
def _process_output(y, n_batch, target_channels, n_channels, stereo_mode, resampler):

    n_batch_int = int(n_batch.item())  # Convert the JIT-Tensor value to a Python int
    target_channels = int(target_channels.item())
    n_channels = int(n_channels.item())
    stereo_mode = bool(stereo_mode.item())

    if stereo_mode:  # stereo_mode is now a Python bool

        n_batch_int = int(n_batch_int / 2)
        y = torch.cat([y[:n_batch_int], y[n_batch_int:]], 1)
    elif target_channels > n_channels:  # target_channels/n_channels are now Python ints
        y = torch.cat(y.chunk(target_channels, 0), 1)
    elif target_channels < n_channels:
        y = y[:, :target_channels]

    if resampler is not None:
        y = resampler.from_model_sampling_rate(y)

    return y


class ScriptedRAVE(nn_tilde.Module):

    def __init__(
        self,
        pretrained: rave.RAVE,
        channels: Optional[int] = None,
        fidelity: float = 0.95,
        target_sr: bool = None,
        prior: prior.Prior = None,
    ) -> None:

        super().__init__()
        self.pqmf = pretrained.pqmf
        self.sr = pretrained.sr
        self.spectrogram = pretrained.spectrogram

        self.resampler = None

        self.input_mode = pretrained.input_mode
        self.output_mode = pretrained.output_mode

        self.n_channels = pretrained.n_channels
        self.target_channels = channels or self.n_channels
        self.stereo_mode = False

        self.dummy_resampler = IdentityModule().to(pretrained.device)  # Initialize a dummy resampler

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = rave.resampler.Resampler(target_sr, self.sr)
                self.sr = target_sr

        # --- FIX: Ensure self.resampler is NOT None ---
        if self.resampler is None:
            self.resampler = self.dummy_resampler

        self.full_latent_size = pretrained.latent_size
        self.is_using_adain = False
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                self.is_using_adain = True
                break
        if self.is_using_adain and (self.n_channels != self.target_channels):
            raise ValueError("AdaIN requires the original number of channels")

        self.register_attribute("learn_target", False)
        self.register_attribute("reset_target", False)
        self.register_attribute("learn_source", False)
        self.register_attribute("reset_source", False)

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        self.latent_size = self.full_latent_size

        if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):

            reduced_latent_size = max(np.argmax(pretrained.fidelity.numpy() > fidelity), 1)
            reduced_latent_size = 2 ** math.ceil(math.log2(reduced_latent_size))

            self.vsa_latent_size = reduced_latent_size

        elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
            self.latent_size = pretrained.encoder.num_quantizers

        elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
            self.latent_size = pretrained.latent_size

        elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
            self.latent_size = pretrained.latent_size - 1

        else:
            raise ValueError(f"Encoder type {pretrained.encoder.__class__.__name__} not supported")

        self.fake_adain = rave.blocks.AdaptiveInstanceNormalization(0)

        self.encoder = pretrained.encoder
        # self.decoder = pretrained.decoder
        self.decoder = HapticDecoderWrapper(pretrained.decoder)

        x_len = 2**14
        x = torch.zeros(1, self.n_channels, x_len)
        z = self.encode(x)
        ratio_encode = x_len // z.shape[-1]

        # --- [NEW VARIABLE: Define the Haptic Downsampling Ratio] ---
        # 441 is the downsampling ratio from 44100 Hz to 100 Hz (44100/100 = 441)
        LATENT_TO_AUDIO_RATIO = ratio_encode
        OUTPUT_RATIO = 1
        LATENT_INPUT_SIZE_FOR_8192_TEST = 8192 // LATENT_TO_AUDIO_RATIO
        SAFE_LATENT_SIZE = 512
        # -----------------------------------------------------------

        # configure encoder
        if (pretrained.input_mode == "pqmf") or (pretrained.output_mode == "pqmf"):
            # scripting fails if cached conv is not initialized
            # --- FIX: Only call self.pqmf if it is not None ---
            if self.pqmf is not None:
                self.pqmf(torch.zeros(1, 1, x_len))
            # --------------------------------------------------

        encode_shape = (pretrained.n_channels, 2**14)

        self.register_method(
            "encode",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=LATENT_TO_AUDIO_RATIO,
            input_labels=["(signal) Channel %d" % d for d in range(1, self.n_channels + 1)],
            output_labels=[f"(signal) Latent dimension {i + 1}" for i in range(self.latent_size)],
        )

        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=1,  # 128 or 256
            out_channels=self.target_channels,
            out_ratio=LATENT_TO_AUDIO_RATIO,
            input_labels=[f"(signal) Latent dimension {i+1}" for i in range(self.latent_size)],
            output_labels=["(signal) Channel %d" % d for d in range(1, self.target_channels + 1)],
            # --- CRITICAL PATCH: Force a valid test buffer size ---
            # test_buffer_size=z.shape[-1],  # Use the same size that passed the encode test
            test_buffer_size=SAFE_LATENT_SIZE,  # Use the same size that passed the encode test
            # ------------------------------------------------------
        )

        self.register_method(
            "forward",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.target_channels,
            out_ratio=OUTPUT_RATIO,
            input_labels=["(signal) Channel %d" % d for d in range(1, self.n_channels + 1)],
            output_labels=["(signal) Channel %d" % d for d in range(1, self.target_channels + 1)],
        )

        # init prior in case
        self._has_prior = False

        if prior is not None:
            self._has_prior = True
            self.prior_module = prior
            self.register_method(
                "prior", in_channels=1, in_ratio=prior.ratio, out_channels=self.latent_size, out_ratio=prior.ratio
            )
        else:
            self._has_prior = False
            self.prior_module = DumbPrior()  # Use DumbPrior as default
            self.register_method(
                "prior",
                in_channels=1,
                in_ratio=self.prior_module.ratio,  # Access ratio from DumbPrior instance
                out_channels=self.latent_size,
                out_ratio=self.prior_module.ratio,
            )
            # self.prior_module = DumbPrior()

    def post_process_latent(self, z):
        raise NotImplementedError

    def pre_process_latent(self, z):
        raise NotImplementedError

    def update_adain(self):
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                m.learn_x.zero_()
                m.learn_y.zero_()

                if self.learn_target[0]:
                    m.learn_y.add_(1)
                if self.learn_source[0]:
                    m.learn_x.add_(1)

                if self.reset_target[0]:
                    m.reset_y()
                if self.reset_source[0]:
                    m.reset_x()

        self.reset_source = (False,)
        self.reset_target = (False,)

    @torch.jit.export
    def set_stereo_mode(self, stereo):
        self.stereo_mode = bool(stereo)

    @torch.jit.export
    def encode(self, x) -> torch.Tensor:

        if self.stereo_mode:
            if self.n_channels == 1:
                x = x[:, 0].unsqueeze(0)
            elif self.n_channels > 2:
                raise RuntimeError("stereo mode is not available when n_channels > 2")

        if self.is_using_adain:
            self.update_adain()

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        batch_size = x.shape[:-2]
        if self.input_mode == "pqmf":
            x = x.reshape(-1, 1, x.shape[-1])
            if self.pqmf is not None:
                x = self.pqmf(x)
            x = x.reshape(batch_size + (-1, x.shape[-1]))

        elif self.input_mode == "mel":
            if self.spectrogram is not None:
                x = self.spectrogram(x)[..., :-1]
                x = torch.log1p(x).reshape(batch_size + (-1, x.shape[-1]))
            else:
                raise RuntimeError("Spectrogram was not initialized")
        z = self.encoder(x)
        z = self.post_process_latent(z)
        return z

    @torch.jit.export
    def decode(self, z, from_forward: bool = False, from_jit: bool = False) -> torch.Tensor:

        if self.is_using_adain and not from_forward:
            self.update_adain()

        y = self.decoder(z)
        # Inlined logic from _process_output - ONLY RUN IF from_forward IS TRUE
        if from_forward:

            # Use self.stereo_mode directly (bool)
            if self.stereo_mode:
                # Note: z.shape[0] is the current batch size before potential stereo splitting
                n_batch_int = z.shape[0] // 2
                y = torch.cat([y[:n_batch_int], y[n_batch_int:]], dim=1)
            # Use self.target_channels and self.n_channels directly (int attributes)
            elif self.target_channels > self.n_channels:
                # Use int attributes directly
                y = torch.cat(y.chunk(self.target_channels, 0), dim=1)
            elif self.target_channels < self.n_channels:
                y = y[:, : self.target_channels]

            # Resampler call - self.resampler is a JIT-scripted module (IdentityModule or Resampler)
            # The check 'if self.resampler is not None:' is technically not needed
            # because self.resampler is always initialized to self.dummy_resampler (IdentityModule)
            # which is a ScriptModule, but we'll include it for clarity if the check was kept in _process_output
            # if self.resampler is not None: # check is no longer needed
            y = self.resampler.from_model_sampling_rate(y)

            return y

        # Original return path when from_forward is False (i.e., not called from JIT's 'forward')
        return y

        # if not from_forward:
        #     return y

        # n_batch_int = z.shape[0]  # is a python int in the JIT's context.
        # n_batch = torch.tensor(n_batch_int, device=z.device)
        # target_channels_tensor = torch.tensor(self.target_channels, device=z.device)
        # n_channels_tensor = torch.tensor(self.n_channels, device=z.device)
        # stereo_mode_tensor = torch.tensor(self.stereo_mode, device=z.device)

        # return _process_output(
        #     y, n_batch, target_channels_tensor, n_channels_tensor, stereo_mode_tensor, self.resampler
        # )

    def forward(self, x):
        return self.decode(self.encode(x), from_forward=True, from_jit=False)

    @torch.jit.export
    def get_learn_target(self) -> bool:
        return self.learn_target[0]

    @torch.jit.export
    def set_learn_target(self, learn_target: bool) -> int:
        self.learn_target = (learn_target,)
        return 0

    @torch.jit.export
    def get_learn_source(self) -> bool:
        return self.learn_source[0]

    @torch.jit.export
    def set_learn_source(self, learn_source: bool) -> int:
        self.learn_source = (learn_source,)
        return 0

    @torch.jit.export
    def get_reset_target(self) -> bool:
        return self.reset_target[0]

    @torch.jit.export
    def set_reset_target(self, reset_target: bool) -> int:
        self.reset_target = (reset_target,)
        return 0

    @torch.jit.export
    def get_reset_source(self) -> bool:
        return self.reset_source[0]

    @torch.jit.export
    def set_reset_source(self, reset_source: bool) -> int:
        self.reset_source = (reset_source,)
        return 0

    @torch.jit.export
    def prior(self, temp: torch.Tensor):
        if self._has_prior:
            return self.prior_module.forward(temp)
        else:
            return torch.tensor(0)


class VariationalScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.encoder.reparametrize(z)[0]
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, : self.latent_size]
        return z

    def pre_process_latent(self, z):
        noise = torch.randn(
            z.shape[0],
            self.full_latent_size - z.shape[1],
            z.shape[-1],
        ).type_as(z)
        z = torch.cat([z, noise], 1)
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        return z


class DiscreteScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.encoder.rvq.encode(z)
        return z.float()

    def pre_process_latent(self, z):
        z = torch.clamp(z, 0, self.encoder.rvq.layers[0].codebook_size - 1).long()
        z = self.encoder.rvq.decode(z)
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation, z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class WasserteinScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return z

    def pre_process_latent(self, z):
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation, z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class SphericalScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return rave.blocks.unit_norm_vector_to_angles(z)

    def pre_process_latent(self, z):
        return rave.blocks.angles_to_unit_norm_vector(z)


class TraceModel(nn.Module):
    def __init__(self, pretrained: prior.Prior, model: rave.RAVE):
        super().__init__()
        pretrained._jit_is_scripting = True
        self.pretrained = pretrained
        self.latent_size = pretrained.latent_size

        x = torch.zeros(1, self.pretrained.n_channels, 2**14)
        z = model.encode(x)
        z = pretrained.post_process_latent(z)
        self.ratio = x.shape[-1] // z.shape[-1]

        self.pretrained.synth = None

        self.register_buffer(
            "previous_step",
            self.pretrained.quantized_normal.encode(torch.zeros(1, self.latent_size, 1)),
        )

        # self.pre_diag_cache = cc.CachedPadding1d(self.latent_size - 1)
        # self.pre_diag_cache(z)

        # if hasattr(self.pre_diag_cache, "pad"):
        #     self.register_buffer("pre_diag_cache_pad", self.pre_diag_cache.pad)
        #     self.pre_diag_cache.pad = self.pre_diag_cache_pad

        self.pre_diag_cache = cc.CachedPadding1d(self.latent_size - 1)

        # --- FIX: Force initialization and register 'pad' explicitly for JIT ---
        # The padding tensor must be created BEFORE TorchScripting
        # Call the module to initialize the internal 'pad' tensor
        z_init = torch.zeros(1, self.latent_size, 2**14 // self.ratio)  # Use a dummy input shape
        self.pre_diag_cache(z_init)

        # Now, self.pre_diag_cache has the 'pad' attribute.
        # Register it as a buffer directly, or re-register a copy.
        if hasattr(self.pre_diag_cache, "pad"):
            # Register the pad tensor as a buffer on the TraceModel
            self.register_buffer("pre_diag_cache_pad", self.pre_diag_cache.pad)
            # The original module's 'pad' must reference this buffer for JIT compatibility
            self.pre_diag_cache.pad = self.pre_diag_cache_pad
        # ------------------------------------------------------------------------

    @torch.jit.ignore  # ADDED
    def step_forward(self, temp):
        # PREDICT NEXT STEP
        x = self.pretrained.forward(self.previous_step)
        x = x / temp
        x = self.pretrained.post_process_prediction(x, argmax=False)
        self.previous_step.copy_(x.clone())

        # DECODE AND SHIFT PREDICTION
        x = self.pretrained.quantized_normal.decode(x)
        # x = self.pre_diag_cache(x)
        x = self.pretrained.diagonal_shift.inverse(x)
        return x

    @torch.jit.ignore  # ADDED
    def forward(self, temp: torch.Tensor):
        x = torch.zeros(
            temp.shape[0],
            self.latent_size,
            temp.shape[-1],
        ).to(temp)

        temp = temp.mean(-1, keepdim=True)
        temp = nn.functional.softplus(temp) / math.log(2)

        for i in range(x.shape[-1]):
            x[..., i : i + 1] = self.step_forward(temp)

        return x


prior_classes = ["VariationalPrior"]


def get_prior_class_from_config():
    prior_class = None
    for cl in prior_classes:
        try:
            gin.get_bindings(cl)
            prior_class = cl
        except:
            pass
    if prior_class is None:
        raise RuntimeError("Could not retrive Prior class from gin config")
    return prior_class


def get_state_dict(RUN, PRIOR):
    state_dict = torch.load(PRIOR, map_location="cpu")["state_dict"]
    for k, v in RUN.state_dict().items():
        state_dict[f"synth.{k}"] = v
    return state_dict


def main(argv):
    cc.use_cached_conv(FLAGS.streaming)

    logging.info("building rave")

    config_file = rave.core.search_for_config(FLAGS.run)
    if config_file is None:
        print("Config file not found in %s" % FLAGS.run)
    gin.parse_config_file(config_file)
    FLAGS.run = rave.core.search_for_run(FLAGS.run)

    pretrained = rave.RAVE()
    if FLAGS.run is not None:
        logging.info("model found : %s" % FLAGS.run)
        checkpoint = torch.load(FLAGS.run, map_location="cpu")
        if FLAGS.ema_weights and "EMA" in checkpoint["callbacks"]:
            pretrained.load_state_dict(
                checkpoint["callbacks"]["EMA"],
                strict=False,
            )
        else:
            pretrained.load_state_dict(
                checkpoint["state_dict"],
                strict=False,
            )
    else:
        logging.error("No checkpoint found")
        exit()
    pretrained.eval()

    pretrained.output_mode = "raw"

    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = VariationalScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
        script_class = DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        script_class = WasserteinScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
        script_class = SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(pretrained.encoder)} " "not supported for export.")

    logging.info("warmup pass")

    x = torch.zeros(1, pretrained.n_channels, 2**14)
    # pretrained(x)
    # 1. Instantiate the Haptic Decoder Wrapper
    haptic_decoder_wrapper_instance = HapticDecoderWrapper(pretrained.decoder)

    # 2. Perform warmup on the wrapper instance (which isolates the haptic path)
    z_dummy = pretrained.encode(x)
    haptic_decoder_wrapper_instance(z_dummy)  # <--- NEW WARMUP CALL

    logging.info("optimize model")

    # parse prior
    prior_scripted = None
    if FLAGS.prior is not None:
        logging.info("loading prior from checkpoint")
        prior_config_file = rave.core.search_for_config(FLAGS.prior)
        if prior_config_file is None:
            print("Config file for prior not found in %s" % FLAGS.prior)
        else:
            gin.clear_config()
            logging.info("prior config file : ", prior_config_file)
            gin.parse_config_file(prior_config_file)
            PRIOR = rave.core.search_for_run(FLAGS.prior)
            logging.info(f"using prior model at {PRIOR}")
            prior_class = get_prior_class_from_config()
            prior_pretrained = getattr(prior, prior_class)(pretrained_vae=pretrained, n_channels=pretrained.n_channels)
            prior_pretrained.load_state_dict(get_state_dict(pretrained, PRIOR))
            prior_scripted_py = TraceModel(prior_pretrained, pretrained)

            # --- FIX: Script the TraceModel separately BEFORE passing it to ScriptedRAVE
            try:
                prior_scripted = torch.jit.script(prior_scripted_py)
                logging.info("TraceModel scripted successfully.")
            except Exception as e:
                logging.error(f"Failed to script TraceModel: {e}")
                raise

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    # We must explicitly script the HapticDecoderWrapper before passing it.
    # The pretrained.decoder is the GeneratorV2 instance.
    haptic_decoder_wrapper_instance = HapticDecoderWrapper(pretrained.decoder)

    # --- CRITICAL FIX 1: Script the HapticDecoderWrapper ---
    try:
        # Script the wrapper instance
        scripted_decoder = torch.jit.script(haptic_decoder_wrapper_instance)
        logging.info("HapticDecoderWrapper successfully scripted.")
    except Exception as e:
        logging.error(f"Failed to script HapticDecoderWrapper: {e}")
        # If this fails, the export cannot proceed.
        raise RuntimeError(f"Failed to script HapticDecoderWrapper: {e}") from e

    # ----------------------------------------------------

    # --- CRITICAL FIX 2: Temporarily patch the pretrained object ---
    # The ScriptedRAVE class needs the original decoder object passed to it,
    # but the HapticDecoderWrapper must use the scripted version.
    # Since ScriptedRAVE calls HapticDecoderWrapper(pretrained.decoder),
    # we need to pass the *scripted* wrapper.
    # Let's replace the decoder attribute on the pretrained object temporarily
    # with our scripted wrapper, allowing ScriptedRAVE to wrap it again
    # but with the correctly traced module. (This is complex due to the RAVE wrapper structure)

    # Simplest approach: Pass the scripted wrapper directly as the decoder when creating ScriptedRAVE,
    # and adjust the ScriptedRAVE __init__ to expect a pre-wrapped/scripted module.

    # Since the ScriptedRAVE __init__ structure is fixed, let's modify the HapticDecoderWrapper
    # to be created outside and then temporarily replace the decoder.

    # The ScriptedRAVE __init__ line is: self.decoder = HapticDecoderWrapper(pretrained.decoder)
    # We must pass the *scripted* version.

    # Since ScriptedRAVE takes the pretrained object, let's temporarily replace the decoder module on it.

    # We create the ScriptedRAVE instance using the original decoder in the pretrained object,
    # but since the HapticDecoderWrapper is correctly designed, the script should have worked.

    # The issue is the double-wrap. Let's force the model's decoder to be the scripted wrapper itself.
    pretrained.decoder = scripted_decoder  # Replace GeneratorV2 with scripted HapticDecoderWrapper
    # ----------------------------------------------------

    logging.info("script model")
    scripted_rave = script_class(
        pretrained=pretrained,
        channels=FLAGS.channels,
        fidelity=FLAGS.fidelity,
        target_sr=FLAGS.sr,
        prior=prior_scripted if prior_scripted is not None else DumbPrior(latent_size=pretrained.latent_size),
    )
    z = scripted_rave.encode(x)
    x = scripted_rave.decode(z)

    logging.info("save model")
    output = FLAGS.output or os.path.dirname(FLAGS.run)
    model_name = FLAGS.name or FLAGS.run.split(os.sep)[-4]
    if FLAGS.streaming:
        model_name += "_streaming"
    model_name += ".ts"

    output = os.path.abspath(output)
    if not os.path.isdir(output):
        os.makedirs(output)

    final_output_path = os.path.join(output, model_name)

    scripted_module = torch.jit.script(scripted_rave)

    torch.jit.save(scripted_module, final_output_path)

    logging.info(f"PyTorch JIT save executed for: {final_output_path}")
    try:
        if pretrained.n_channels <= 2:
            # test stereo mode for VST export
            scripted_rave.set_stereo_mode(True)
            z_vst_input = torch.zeros(2, scripted_rave.full_latent_size, z.shape[-1])
            out = scripted_rave.decode(z_vst_input)
            assert out.shape[1] == 2, "model output is not stereo"
            logging.info(f"this model seems compatible with the RAVE vst.")
    except Exception as e:
        logging.warning(f"this model will not work with the RAVE VST. \n Caught error : %s" % e)

    logging.info(f"all good ! model exported to {os.path.join(output, model_name)}")


if __name__ == "__main__":
    app.run(main)
