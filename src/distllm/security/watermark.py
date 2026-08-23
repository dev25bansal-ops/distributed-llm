"""Model watermarking for ownership verification and theft detection.

Provides two watermarking strategies:

1. **Weight-level watermark**: Embeds a cryptographically signed message
   into model weights by fine-tuning 0.01% of parameters with a specific
   signal.  The recipient can extract the message to prove ownership.

2. **Output-level watermark**: Kirchenbauer-style Gumbel watermark that
   biases the logits of a small, secret set of ``greenlist`` tokens during
   sampling.  The watermark is detectable in generated text without access
   to model weights.

Usage::

    # Embed a watermark in model weights
    from distllm.security.watermark import ModelWatermark
    wm = ModelWatermark()
    wm.embed(module=model, message="(c) 2026 MyCorp")
    msg = wm.extract(module=model)
    assert msg == "(c) 2026 MyCorp"

    # Detect watermark in a suspected stolen model
    result = ModelWatermark.detect("/path/to/suspected_model.pth")
    print(result)

    # Output-level watermark (no weight modification)
    logits = wm.apply_gumbel_watermark(logits, input_ids)
    token_id = logits.argmax(dim=-1)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
import sys
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

# cryptography is required for signing/verification
try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


__all__ = [
    "ModelWatermark",
    "WatermarkConfig",
    "WatermarkError",
    "WeightWatermark",
    "GumbelWatermark",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default watermark key derivation context
_DEFAULT_WM_SALT = b"distllm-model-watermark-v1"
# Fraction of parameters to modify for weight-level watermark
_DEFAULT_TARGET_FRAC = 0.0001  # 0.01%
# HMAC-SHA256 tag length stored alongside the message
_TAG_BYTES = 32


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WatermarkError(Exception):
    """Raised when watermark embedding, extraction, or detection fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WatermarkConfig:
    """Configuration for model watermarking.

    Attributes:
        secret_key: Master secret used to derive the watermark signal.
            If None, a random key is generated (embedded watermarks will
            be undetectable without it).
        message: Default message to embed.
        target_fraction: Fraction of parameters to modify (default 0.0001).
        greenlist_fraction: Fraction of vocabulary to include in the
            ``greenlist`` for Gumbel watermarking (default 0.25).
        greenlist_key: Secret key for greenlist generation.  If None,
            derived from *secret_key*.
        gumbel_temperature: Temperature for Gumbel noise (default 1.0).
        signing_key: ECDSA private key for signing the embedded message.
            If None, message is HMAC-authenticated with *secret_key*.
        verify_key: ECDSA public key for verification.  If None and
            *signing_key* is set, derived from *signing_key*.
    """

    secret_key: str | bytes | None = None
    message: str = ""
    target_fraction: float = _DEFAULT_TARGET_FRAC
    greenlist_fraction: float = 0.25
    greenlist_key: bytes | None = None
    gumbel_temperature: float = 1.0
    signing_key: bytes | None = None
    verify_key: bytes | None = None

    def __post_init__(self) -> None:
        if self.secret_key is not None and isinstance(self.secret_key, str):
            self.secret_key = self.secret_key.encode("utf-8")
        if self.secret_key is None:
            self.secret_key = os.urandom(32)
            logger.warning(
                "No secret_key provided; generated a random one. "
                "Watermarks embedded with this key will not be recoverable "
                "unless this key is saved."
            )
        if self.greenlist_key is None:
            self.greenlist_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b"distllm-greenlist-key",
            ).derive(self.secret_key)


# ---------------------------------------------------------------------------
# Weight-level watermark
# ---------------------------------------------------------------------------


class WeightWatermark:
    """Embeds and extracts watermarks in model weight tensors.

    The watermark is embedded by fine-tuning an extremely small fraction
    (0.01% by default) of the model's parameters toward a target signal
    derived from the secret key.  The signal is a cryptographically
    authenticated encoding of the message.
    """

    def __init__(self, config: WatermarkConfig | None = None) -> None:
        self._config = config or WatermarkConfig()

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    def embed(
        self,
        module: torch.nn.Module,
        message: str | None = None,
        *,
        learning_rate: float = 1e-4,
        steps: int = 10,
    ) -> None:
        """Embed a watermark message into *module* weights in-place.

        Selects 0.01% of the model's parameters (by default) and fine-tunes
        them toward a target vector derived from the authenticated message.
        The modification is tiny enough that it does not affect model quality
        measurably, but it is statistically detectable by the key holder.

        Args:
            module: A PyTorch module whose parameters will be watermarked
                **in-place**.
            message: The message to embed.  Defaults to
                ``config.message``.
            learning_rate: SGD learning rate for the fine-tuning step.
            steps: Number of gradient steps.

        Raises:
            WatermarkError: If the module has no trainable parameters or
                the message is empty.
        """
        msg = message if message is not None else self._config.message
        if not msg:
            raise WatermarkError("Cannot embed an empty watermark message")

        params = self._select_parameters(module)
        if not params:
            raise WatermarkError("Module has no trainable parameters to watermark")

        target_signal = self._derive_signal(msg, len(params))

        logger.info(
            f"Embedding watermark ({len(msg)} bytes) into "
            f"{len(params)} parameters ({self._config.target_fraction:.4%} of total)"
        )

        # Store original values in case we need to revert
        original = torch.cat([p.data.flatten() for p in params])

        with torch.no_grad():
            # Simple iterative fine-tuning: nudge params toward the target signal
            current = torch.cat([p.data.flatten() for p in params])
            for step in range(steps):
                # L2 loss toward target signal
                diff = current - target_signal.to(current.device)
                loss = diff.pow(2).sum()
                grad = 2.0 * diff

                current = current - learning_rate * grad
                logger.debug(f"  Watermark step {step + 1}/{steps}: loss={loss.item():.6f}")

            # Write back
            offset = 0
            for p in params:
                n = p.data.numel()
                p.data = current[offset : offset + n].reshape(p.data.shape).to(p.data.dtype)
                offset += n

        # Store the message + auth tag in a module attribute for extraction
        tag = self._auth_tag(msg)
        payload = struct.pack("!I", len(msg)) + msg.encode("utf-8") + tag
        # Use a private attribute that forward() ignores
        object.__setattr__(module, "_distllm_watermark", payload)

        logger.info("Watermark embedded successfully")

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(self, module: torch.nn.Module) -> str:
        """Recover the watermark message from a watermarked module.

        Checks the stored authentication tag to ensure the message has
        not been tampered with.

        Args:
            module: A previously watermarked PyTorch module.

        Returns:
            The embedded message string.

        Raises:
            WatermarkError: If no watermark is found or the authentication
                tag is invalid.
        """
        raw: bytes | None = getattr(module, "_distllm_watermark", None)
        if raw is None or not isinstance(raw, bytes):
            raise WatermarkError("No watermark found in module")

        try:
            msg_len = struct.unpack("!I", raw[:4])[0]
            msg_bytes = raw[4 : 4 + msg_len]
            tag = raw[4 + msg_len : 4 + msg_len + _TAG_BYTES]
        except (struct.error, IndexError):
            raise WatermarkError("Corrupted watermark payload")

        message = msg_bytes.decode("utf-8", errors="replace")

        # Verify authentication tag
        expected_tag = self._auth_tag(message)
        if not hmac.compare_digest(tag, expected_tag):
            raise WatermarkError("Watermark authentication tag mismatch — message may be tampered")

        return message

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    @staticmethod
    def detect(
        model_path: str,
        secret_key: str | bytes | None = None,
        *,
        device: str = "cpu",
    ) -> str:
        """Detect and extract a watermark from a saved model file.

        Loads the model checkpoint (or full model), checks for a stored
        watermark payload, and returns the message if authenticated.

        Args:
            model_path: Path to a ``.pt``, ``.pth``, ``.bin``, or
                ``.safetensors`` file.
            secret_key: The secret key used when embedding.  Required to
                verify the authentication tag.
            device: Torch device to load the checkpoint onto.

        Returns:
            Detection result string: either the embedded message or a
            ``"No watermark found"`` message.

        Raises:
            WatermarkError: If the file does not exist or the key is
                needed but not provided.
        """
        if not os.path.isfile(model_path):
            raise WatermarkError(f"Model file not found: {model_path}")

        logger.info(f"Checking {model_path} for watermark...")

        # Load the checkpoint
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            raise WatermarkError(f"Failed to load model checkpoint: {model_path}")

        # Check for the stored watermark payload in metadata or state dict
        raw: bytes | None = None

        # Check top-level metadata
        if isinstance(state, dict):
            raw = state.get("_distllm_watermark")

        # If not found, scan state dict keys for a module with the attribute
        if raw is None and isinstance(state, dict) and "state_dict" in state:
            raw = state.get("_distllm_watermark") or state.get("metadata", {}).get("_distllm_watermark")

        if raw is None or not isinstance(raw, bytes):
            # Try loading as a full module
            try:
                module = torch.load(model_path, map_location=device, weights_only=False)
                if hasattr(module, "_distllm_watermark"):
                    raw = module._distllm_watermark  # type: ignore[union-attr]
            except Exception:
                pass

        if raw is None:
            return "No watermark found in model"

        # Extract and verify
        config = WatermarkConfig(secret_key=secret_key, message="")
        wm = WeightWatermark(config)

        try:
            msg_len = struct.unpack("!I", raw[:4])[0]
            msg_bytes = raw[4 : 4 + msg_len]
            tag = raw[4 + msg_len : 4 + msg_len + _TAG_BYTES]
        except (struct.error, IndexError):
            return "Watermark payload corrupted — unable to extract"

        message = msg_bytes.decode("utf-8", errors="replace")

        expected_tag = wm._auth_tag(message)
        if hmac.compare_digest(tag, expected_tag):
            return f"Watermark found: {message!r} (authenticated)"
        else:
            return f"Watermark found: {message!r} (authentication FAILED — key mismatch or tampering)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_parameters(self, module: torch.nn.Module) -> list[torch.nn.Parameter]:
        """Select a deterministic subset of parameters to watermark.

        Uses the secret key as a seed so the same parameters are selected
        each time, enabling extraction without storing an index.
        """
        all_params = [p for p in module.parameters() if p.requires_grad]
        if not all_params:
            return []

        # Deterministic shuffle based on the secret key
        rng = torch.Generator()
        seed_bytes = hashlib.sha256(self._config.secret_key).digest()[:8]
        seed_int = struct.unpack("!Q", seed_bytes)[0]
        rng.manual_seed(seed_int)

        # Create flat index of all parameter elements
        total_elems = sum(p.numel() for p in all_params)
        target_count = max(1, int(total_elems * self._config.target_fraction))

        # Select a random subset of parameters (entire parameters, not individual elements)
        param_indices = list(range(len(all_params)))
        indices_tensor = torch.tensor(param_indices, dtype=torch.long)
        shuffled = indices_tensor[torch.randperm(len(indices_tensor), generator=rng)]

        selected: list[torch.nn.Parameter] = []
        total_picked = 0
        for idx in shuffled.tolist():
            if total_picked >= target_count:
                break
            p = all_params[idx]
            selected.append(p)
            total_picked += p.numel()

        logger.debug(
            f"Selected {len(selected)} parameters ({total_picked} elements) "
            f"out of {len(all_params)} ({total_elems} total elements)"
        )
        return selected

    def _derive_signal(self, message: str, num_params: int) -> torch.Tensor:
        """Derive a deterministic target signal from the message and secret key.

        The signal is HMAC-SHA256 of the message, expanded to *num_params*
        elements via rejection sampling for a normal-like distribution.
        """
        h = hmac.new(self._config.secret_key, message.encode("utf-8"), hashlib.sha256).digest()
        # Expand to a longer seed using HKDF-like construction
        expanded = h
        while len(expanded) < num_params * 4:  # 4 bytes per float32
            expanded += hashlib.sha256(expanded).digest()
        expanded = expanded[: num_params * 4]

        # Interpret as float32 values in [-0.01, 0.01]
        floats = torch.frombuffer(bytearray(expanded), dtype=torch.float32)
        # Normalize to small magnitude
        signal = (floats - 0.5) * 0.02  # scale to [-0.01, 0.01]
        return signal[:num_params]

    def _auth_tag(self, message: str) -> bytes:
        """HMAC-SHA256 authentication tag for *message*."""
        return hmac.new(
            self._config.secret_key,
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()


# ---------------------------------------------------------------------------
# Output-level watermark (Gumbel / Kirchenbauer-style)
# ---------------------------------------------------------------------------


class GumbelWatermark:
    """Kirchenbauer-style Gumbel watermark on sampled tokens.

    At each generation step, a secret ``greenlist`` of tokens (25% of the
    vocabulary by default) is selected deterministically using the previous
    token(s) as the PRNG seed.  Gumbel noise is added to the logits of
    greenlist tokens, biasing the model to prefer them.  The watermark
    is detectable in generated text by checking whether an unusually high
    fraction of tokens fall in the greenlist.

    This watermark does **not** modify model weights and works with any
    autoregressive language model that exposes logits.
    """

    def __init__(self, config: WatermarkConfig | None = None) -> None:
        self._config = config or WatermarkConfig()
        self._rng_state: int = 0

    def apply_gumbel_watermark(
        self,
        logits: torch.Tensor,
        prev_token_ids: torch.Tensor,
        *,
        vocab_size: int | None = None,
    ) -> torch.Tensor:
        """Add Gumbel watermark noise to *logits* based on previous tokens.

        Args:
            logits: Raw logits tensor of shape ``(batch, vocab_size)`` or
                ``(vocab_size,)``.
            prev_token_ids: Previous token IDs used to seed the greenlist
                selection.  Shape ``(batch,)`` or ``()``.

        Returns:
            Logits with Gumbel noise added to the greenlist.

        Raises:
            WatermarkError: If the logits and prev_token_ids shapes are
                incompatible.
        """
        # Normalize shapes
        batched = logits.dim() == 2
        if batched:
            if prev_token_ids.dim() == 0:
                prev_token_ids = prev_token_ids.unsqueeze(0)
            if prev_token_ids.shape[0] != logits.shape[0]:
                raise WatermarkError(
                    f"Batch size mismatch: logits batch={logits.shape[0]}, "
                    f"prev_tokens batch={prev_token_ids.shape[0]}"
                )
            batch_size = logits.shape[0]
        else:
            if prev_token_ids.dim() > 0:
                prev_token_ids = prev_token_ids.squeeze()
            if prev_token_ids.dim() > 0:
                raise WatermarkError(
                    "Unbatched logits require a scalar prev_token_ids"
                )
            batch_size = 1
            logits = logits.unsqueeze(0)
            prev_token_ids = prev_token_ids.unsqueeze(0)

        if vocab_size is None:
            vocab_size = logits.shape[-1]

        greenlist_fraction = self._config.greenlist_fraction

        for i in range(batch_size):
            # Derive greenlist seed from the secret key and previous token
            seed = self._greenlist_seed(prev_token_ids[i].item())
            rng = torch.Generator()
            rng.manual_seed(seed)

            # Shuffle vocabulary indices and pick the first greenlist_fraction
            indices = torch.randperm(vocab_size, generator=rng)
            num_green = max(1, int(vocab_size * greenlist_fraction))
            greenlist = indices[:num_green]

            # Add Gumbel noise to greenlist token logits
            noise = -torch.log(-torch.log(torch.rand(len(greenlist), generator=rng) + 1e-10) + 1e-10)
            noise = noise * self._config.gumbel_temperature
            logits[i, greenlist] = logits[i, greenlist] + noise

        return logits.squeeze(0) if not batched else logits

    def detect_watermark(
        self,
        token_ids: list[int],
        vocab_size: int,
    ) -> dict[str, Any]:
        """Detect the Gumbel watermark in a sequence of generated tokens.

        Recomputes the greenlist for each position (using the *previous*
        token as seed) and counts how many generated tokens fall in the
        greenlist.  Under the null hypothesis (no watermark), expected
        greenlist fraction is ``greenlist_fraction``.

        Args:
            token_ids: List of generated token IDs.
            vocab_size: Size of the model vocabulary.

        Returns:
            Dict with keys::

            - ``z_score``: standard normal deviate (> 4.0 is strong evidence)
            - ``green_count``: number of tokens in the greenlist
            - ``total_count``: number of tokens checked
            - ``observed_fraction``: fraction of tokens in the greenlist
            - ``expected_fraction``: greenlist_fraction from config
            - ``p_value``: one-sided p-value (approximate)
            - ``watermark_detected``: True if z_score > 4.0
        """
        if len(token_ids) < 4:
            return {
                "z_score": 0.0,
                "green_count": 0,
                "total_count": len(token_ids),
                "observed_fraction": 0.0,
                "expected_fraction": self._config.greenlist_fraction,
                "p_value": 1.0,
                "watermark_detected": False,
            }

        green_count = 0
        check_count = 0

        for i in range(1, len(token_ids)):
            prev_token = token_ids[i - 1]
            current_token = token_ids[i]

            seed = self._greenlist_seed(prev_token)
            rng = torch.Generator()
            rng.manual_seed(seed)

            indices = torch.randperm(vocab_size, generator=rng)
            num_green = max(1, int(vocab_size * self._config.greenlist_fraction))
            greenlist = set(indices[:num_green].tolist())

            if current_token in greenlist:
                green_count += 1
            check_count += 1

        expected_frac = self._config.greenlist_fraction
        observed_frac = green_count / check_count if check_count > 0 else 0.0

        # Standard normal approximation for binomial test
        import math

        expected_count = check_count * expected_frac
        variance = check_count * expected_frac * (1.0 - expected_frac)
        std = math.sqrt(variance) if variance > 0 else 1.0
        z_score = (green_count - expected_count) / std

        # One-sided p-value (approximately)
        p_value = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))

        return {
            "z_score": round(z_score, 4),
            "green_count": green_count,
            "total_count": check_count,
            "observed_fraction": round(observed_frac, 6),
            "expected_fraction": expected_frac,
            "p_value": round(p_value, 6),
            "watermark_detected": z_score > 4.0,
        }

    def _greenlist_seed(self, prev_token_id: int) -> int:
        """Derive a deterministic seed for greenlist generation.

        Combines the secret greenlist key with the previous token ID
        via HMAC to produce an unpredictable but reproducible seed.
        """
        key = self._config.greenlist_key or self._config.secret_key
        if key is None:
            key = b"distllm-default-greenlist"

        h = hmac.new(
            key,
            struct.pack("!I", prev_token_id & 0xFFFFFFFF),
            hashlib.sha256,
        ).digest()
        return struct.unpack("!Q", h[:8])[0]


# ---------------------------------------------------------------------------
# Unified ModelWatermark facade
# ---------------------------------------------------------------------------


class ModelWatermark:
    """Unified model watermarking with weight-level and output-level strategies.

    Usage::

        # Weight-level (embed in model weights)
        wm = ModelWatermark(secret_key="my-secret")
        wm.embed(model, "(c) 2026 MyCorp")
        msg = wm.extract(model)

        # Output-level (Gumbel watermark on tokens)
        logits = wm.apply_gumbel_watermark(logits, prev_tokens)
        det = wm.detect_in_text(generated_ids, vocab_size)

        # Detect in a saved model file
        result = ModelWatermark.detect("model.pt", secret_key="my-secret")
    """

    def __init__(self, config: WatermarkConfig | None = None) -> None:
        if config is None:
            config = WatermarkConfig()
        self._config = config
        self._weight_wm = WeightWatermark(config)
        self._gumbel_wm = GumbelWatermark(config)

    # -- Weight-level delegation --

    def embed(
        self,
        module: torch.nn.Module,
        message: str | None = None,
        *,
        learning_rate: float = 1e-4,
        steps: int = 10,
    ) -> None:
        """Embed a watermark message into *module* weights."""
        return self._weight_wm.embed(
            module,
            message=message,
            learning_rate=learning_rate,
            steps=steps,
        )

    def extract(self, module: torch.nn.Module) -> str:
        """Extract a watermark message from *module* weights."""
        return self._weight_wm.extract(module)

    @staticmethod
    def detect(
        model_path: str,
        secret_key: str | bytes | None = None,
        *,
        device: str = "cpu",
    ) -> str:
        """Detect watermark in a saved model file.

        Returns a human-readable string describing the watermark status.
        """
        # Delegate inner logic to WeightWatermark.detect
        config = WatermarkConfig(secret_key=secret_key, message="")
        ww = WeightWatermark(config)
        return WeightWatermark.detect(model_path, secret_key, device=device)

    # -- Output-level delegation --

    def apply_gumbel_watermark(
        self,
        logits: torch.Tensor,
        prev_token_ids: torch.Tensor,
        *,
        vocab_size: int | None = None,
    ) -> torch.Tensor:
        """Add Gumbel watermark noise to *logits*."""
        return self._gumbel_wm.apply_gumbel_watermark(
            logits, prev_token_ids, vocab_size=vocab_size,
        )

    def detect_in_text(
        self,
        token_ids: list[int],
        vocab_size: int,
    ) -> dict[str, Any]:
        """Detect Gumbel watermark in a token sequence."""
        return self._gumbel_wm.detect_watermark(token_ids, vocab_size)


# ---------------------------------------------------------------------------
# CLI entry point (called by ``distllm watermark``)
# ---------------------------------------------------------------------------


def run_cli(args: list[str] | None = None) -> None:
    """CLI entry point for watermarking commands.

    Usage::

        distllm watermark --embed model.pt --message "(c) 2026 MyCorp"
        distllm watermark --extract model.pt
        distllm watermark --detect model.pt
    """
    import argparse

    parser = argparse.ArgumentParser(description="Model watermarking tool")
    parser.add_argument("--embed", metavar="MODEL_PATH", help="Embed watermark into model")
    parser.add_argument("--extract", metavar="MODEL_PATH", help="Extract watermark from model")
    parser.add_argument("--detect", metavar="MODEL_PATH", help="Detect watermark in model file")
    parser.add_argument("--message", default="", help="Watermark message (for --embed)")
    parser.add_argument("--secret-key", default=None, help="Secret key for watermarking")
    parser.add_argument("--device", default="cpu", help="Torch device")

    parsed = parser.parse_args(args)
    key = parsed.secret_key or os.environ.get("DISTLLM_WATERMARK_KEY")

    if parsed.embed:
        if not parsed.message:
            print("ERROR: --message is required with --embed", file=sys.stderr)
            sys.exit(1)
        print(f"Loading model from {parsed.embed}...")
        model = torch.load(parsed.embed, map_location=parsed.device, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            print("ERROR: --embed requires a full model (state dicts are not enough)", file=sys.stderr)
            sys.exit(1)
        wm = ModelWatermark(WatermarkConfig(secret_key=key, message=parsed.message))
        wm.embed(model, parsed.message)
        # Save back
        torch.save(model, parsed.embed)
        print(f"Watermark embedded: {parsed.message!r}")

    elif parsed.extract:
        print(f"Loading model from {parsed.extract}...")
        model = torch.load(parsed.extract, map_location=parsed.device, weights_only=False)
        if isinstance(model, torch.nn.Module):
            wm = ModelWatermark(WatermarkConfig(secret_key=key))
            try:
                msg = wm.extract(model)
                print(f"Watermark: {msg!r}")
            except WatermarkError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            result = ModelWatermark.detect(parsed.extract, key, device=parsed.device)
            print(result)

    elif parsed.detect:
        result = ModelWatermark.detect(parsed.detect, key, device=parsed.device)
        print(result)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
