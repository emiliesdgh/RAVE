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
    def forward(self, x: torch.Tensor):
        return x


# --- [UPDATED MODULE: Haptic Decoder Wrapper for JIT Compatibility] ---
class HapticDecoderWrapper(nn.Module):
    """
    Wraps GeneratorV2 to safely return only the haptic prediction for JIT/nn_tilde.
    Handles cases where the generator returns (audio, haptic) or just (haptic).
    """

    def __init__(self, generator_v2_decoder: nn.Module):
        super().__init__()
        self.generator = generator_v2_decoder

    def forward(self, z):
        # The generator returns either:
        # 1. A tuple: (y_high_rate, haptic_pred) if it's the full-featured version
        # 2. A single Tensor: (haptic_pred) if the output was simplified during training/export

        output = self.generator(z)

        if isinstance(output, tuple):
            # Case 1: Generator returned (audio, haptic). We take the second element (haptic).
            haptic_pred = output[1]
        else:
            # Case 2: Generator returned only the haptic tensor.
            haptic_pred = output

        return haptic_pred


# -----------------------------------------------------------------
# Use torch.jit.ignore to prevent the tracer from seeing this complex channel mapping logic
@torch.jit.ignore
def _process_output(y, n_batch, target_channels, n_channels, stereo_mode, resampler):
    # NOTE: n_batch is a single-element Tensor passed from decode. Convert to Python int.
    n_batch_int = int(n_batch.item())

    if stereo_mode:
        # The haptic wrapper returns 1-channel output, but the stereo logic expects two.
        # This path is likely broken for haptic models, but we keep the logic intact
        # to match the original RAVE structure, protecting it with jit.ignore.
        n_batch_int = int(n_batch_int / 2)
        y = torch.cat([y[:n_batch_int], y[n_batch_int:]], 1)
    elif target_channels > n_channels:
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

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = rave.resampler.Resampler(target_sr, self.sr)
                self.sr = target_sr

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

        # --- [MODIFIED LOGIC] ---
        # 1. Start with the full latent size, which is what the decoder expects as max input.
        self.latent_size = self.full_latent_size

        # 2. Re-calculate the PCA-reduced size, but ONLY use it if it's smaller.
        #    This value is primarily for metadata/VSA.

        if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
            # Calculate the PCA-reduced latent size based on fidelity
            reduced_latent_size = max(np.argmax(pretrained.fidelity.numpy() > fidelity), 1)
            reduced_latent_size = 2 ** math.ceil(math.log2(reduced_latent_size))

            # Store this *reduced* size in a new attribute.
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

        # --- CRITICAL FIX: TRACE THE DECODER FIRST ---
        # Tracing bypasses the type annotation conflict in GeneratorV2 (rave/blocks.py:860)
        # by creating an executable graph from a sample input (z_sample).
        try:
            downsampling_ratio = pretrained.encoder.downsampling_ratio
        except AttributeError:
            # Fallback for older RAVE versions or non-standard configurations
            # The most common downsampling ratio for the latent space is 256.
            downsampling_ratio = 256

        z_len = 2**14 // downsampling_ratio
        # Use map_location to ensure device compatibility (assuming cpu)
        z_sample = torch.zeros(1, pretrained.latent_size, z_len, device=torch.device("cpu"))
        traced_decoder = torch.jit.trace(pretrained.decoder.to(torch.device("cpu")), (z_sample,), strict=False)

        self.encoder = pretrained.encoder
        self.decoder = HapticDecoderWrapper(traced_decoder)  # Pass the TRACED decoder

        x_len = 2**14
        x = torch.zeros(1, self.n_channels, x_len)
        z = self.encode(x)
        ratio_encode = x_len // z.shape[-1]

        # --- [NEW VARIABLE: Define the Haptic Downsampling Ratio] ---
        HAPTIC_DOWNSAMPLING_RATIO = 441
        HACKY_OUT_RATIO = ratio_encode * 2
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
            out_ratio=ratio_encode,
            input_labels=["(signal) Channel %d" % d for d in range(1, self.n_channels + 1)],
            output_labels=[f"(signal) Latent dimension {i + 1}" for i in range(self.latent_size)],
        )

        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=ratio_encode,  # 128 or 256
            out_channels=self.target_channels,
            out_ratio=HACKY_OUT_RATIO,
            input_labels=[f"(signal) Latent dimension {i+1}" for i in range(self.latent_size)],
            output_labels=["(signal) Channel %d" % d for d in range(1, self.target_channels + 1)],
        )

        self.register_method(
            "forward",
            in_channels=self.n_channels,
            in_ratio=1,
            out_channels=self.target_channels,
            out_ratio=HACKY_OUT_RATIO,
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
            self.prior_module = DumbPrior()

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
    def encode(self, x):
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
                raise RuntimeError()
        z = self.encoder(x)
        z = self.post_process_latent(z)
        return z

    @torch.jit.export
    def decode(self, z, from_forward: bool = False, from_jit: bool = False):
        # 1. Update Adain if needed (JIT-safe)
        if self.is_using_adain and not from_forward:
            self.update_adain()

        # 2. Call the decoder wrapper (JIT-safe)
        y = self.decoder(z)

        # If called directly (e.g., by nn_tilde/JIT trace), return the simple output
        if not from_forward:
            return y

        # 3. Restore the complex logic for the full forward pass using the global function
        n_batch_int = z.shape[0]  # Get Python int value for batch size

        # CRITICAL FIX: Explicitly convert the Python int batch size to a Tensor
        # This satisfies the JIT requirement for a dynamic value derived from 'z'.
        n_batch_tensor = torch.tensor(n_batch_int, device=z.device)

        # Pass values as they are, relying on jit.ignore to allow Python types for config.
        return _process_output(
            y, n_batch_tensor, self.target_channels, self.n_channels, self.stereo_mode, self.resampler
        )

    def forward(self, x):
        return self.decode(self.encode(x), from_forward=True)

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
        z = z[:, : self.vsa_latent_size]  # Use the reduced size here
        # or z = z[:, : self.latent_size]
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

        # self.register_buffer(
        #     "forward_params",
        #     torch.tensor([1, self.ratio, self.latent_size, self.ratio]),
        # )

        self.pretrained.synth = None

        self.register_buffer(
            "previous_step",
            self.pretrained.quantized_normal.encode(torch.zeros(1, self.latent_size, 1)),
        )

        self.pre_diag_cache = cc.CachedPadding1d(self.latent_size - 1)
        self.pre_diag_cache(z)

        # --- CRITICAL FIX: Keep the internal buffer fix ---
        if hasattr(self.pre_diag_cache, "pad"):
            self.register_buffer("pre_diag_cache_pad", self.pre_diag_cache.pad)
            self.pre_diag_cache.pad = self.pre_diag_cache_pad
        # ------------------------------------------------------------------------------------

    @torch.jit.ignore  # ADDED
    def step_forward(self, temp):
        # PREDICT NEXT STEP
        x = self.pretrained.forward(self.previous_step)
        x = x / temp
        x = self.pretrained.post_process_prediction(x, argmax=False)
        self.previous_step.copy_(x.clone())

        # DECODE AND SHIFT PREDICTION
        x = self.pretrained.quantized_normal.decode(x)
        x = self.pre_diag_cache(x)
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

    # Force output mode to 'raw' (for haptic output)
    # and nullify PQMF for decoding side (no need for PQMF inverse on haptics)
    pretrained.output_mode = "raw"
    # -----------------------------------------------------------------------

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
            prior_scripted = TraceModel(prior_pretrained, pretrained)

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

    logging.info("script model")
    scripted_rave = script_class(
        pretrained=pretrained,
        channels=FLAGS.channels,
        fidelity=FLAGS.fidelity,
        target_sr=FLAGS.sr,
        prior=prior_scripted,
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
    # scripted_rave.export_to_ts(os.path.join(output, model_name))  # Problematic line
    final_output_path = os.path.join(output, model_name)
    # 1. Script the module explicitly (the previous log messages confirm this works)
    scripted_module = torch.jit.script(scripted_rave)

    # 2. Save the scripted module using the native PyTorch function (less prone to nn_tilde issues)
    torch.jit.save(scripted_module, final_output_path)

    # 3. Add a log to confirm the PyTorch save command executed
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
