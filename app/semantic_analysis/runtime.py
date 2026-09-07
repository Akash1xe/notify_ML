from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
import re
from collections.abc import Callable
from typing import Protocol, Sequence

from PIL import Image, ImageOps

from app.core.config import AppSettings
from app.core.exceptions import (
    VLMCancelledError,
    VLMImageInputError,
    VLMOutOfMemoryError,
    VLMUnsupportedConfigurationError,
)
from app.semantic_analysis.hardware import inspect_hardware
from app.semantic_analysis.models import (
    HardwareCapability,
    VLMGenerationResult,
    VLMRuntimeMetadata,
    VLGenerationConfig,
    VLM_RUNTIME_ALGORITHM_VERSION,
)
from app.video_analysis.fingerprints import stable_hash

CancelCheck = Callable[[], bool]

MODEL_TIERS = {
    "2b": "Qwen/Qwen3-VL-2B-Instruct",
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
}


class VisionLanguageModelAdapter(Protocol):
    def load(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        generation_config: VLGenerationConfig,
        cancel_check: CancelCheck | None = None,
    ) -> tuple[str, int | None]: ...
    def unload(self) -> None: ...
    def runtime_versions(self) -> tuple[str | None, str | None]: ...


class FakeVisionLanguageModelAdapter:
    def __init__(self, responses: list[str] | None = None, *, responder=None, fail_load: Exception | None = None) -> None:
        self.responses = responses
        self.responder = responder
        self.fail_load = fail_load
        self.load_calls = 0
        self.generate_calls = 0
        self._loaded = False

    def load(self) -> None:
        self.load_calls += 1
        if self.fail_load:
            raise self.fail_load
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, images, prompt, generation_config, cancel_check=None):
        if cancel_check and cancel_check():
            raise VLMCancelledError("Semantic inference cancelled")
        if not self._loaded:
            raise RuntimeError("fake adapter not loaded")
        index = self.generate_calls
        self.generate_calls += 1
        if self.responder is not None:
            text = self.responder(prompt, index)
        elif self.responses:
            text = self.responses[min(index, len(self.responses) - 1)]
        else:
            match = re.search(r"Candidate ID:\s*(\d+)", prompt)
            candidate_id = int(match.group(1)) if match else 1
            text = (
                '{"candidate_id":%d,"content_type":"OTHER","completion_state":"COMPLETE",'
                '"completion_score":0.9,"visual_change_state":"SETTLED",'
                '"educational_usefulness":"HIGH","usefulness_score":0.9,'
                '"transition_probability":0.05,"has_meaningful_visual_content":true,'
                '"teacher_still_writing_likely":false,"transcript_visual_consistency":"UNCERTAIN",'
                '"confidence":0.8,"reason_codes":["NEXT_SIMILAR_TO_CURRENT"]}' % candidate_id
            )
        return text, len(text.split())

    def unload(self) -> None:
        self._loaded = False

    def runtime_versions(self):
        return "fake", "fake"


class QwenRuntimeManager:
    def __init__(
        self,
        settings: AppSettings,
        *,
        hardware: HardwareCapability | None = None,
        adapter_factory=None,
    ) -> None:
        self._settings = settings
        self._hardware = hardware or inspect_hardware()
        self._adapter_factory = adapter_factory
        self._adapter: VisionLanguageModelAdapter | None = None
        self._load_lock = threading.RLock()
        self._inference_semaphore = threading.Semaphore(settings.max_concurrent_vlm_inferences)
        self._fallback_used = False
        self._fallback_reason: str | None = None
        self._resolved_model = self._configured_primary_model()
        self._resolved_device = self._resolve_device()
        self._resolved_dtype = self._resolve_dtype(self._resolved_device)
        self._validate_quantization(self._resolved_device)

    def _configured_primary_model(self) -> str:
        return self._settings.qwen_vl_model_name.strip() or MODEL_TIERS[self._settings.qwen_vl_model_tier]

    def _fallback_model(self) -> str:
        return MODEL_TIERS[self._settings.qwen_vl_fallback_model_tier]

    def _minimum_cuda_gb(self, tier: str) -> float:
        return {
            "2b": self._settings.qwen_vl_min_cuda_memory_gb_2b,
            "4b": self._settings.qwen_vl_min_cuda_memory_gb_4b,
            "8b": self._settings.qwen_vl_min_cuda_memory_gb_8b,
        }[tier]

    def _resolve_device(self) -> str:
        configured = self._settings.qwen_vl_device
        if configured == "cpu":
            return "cpu"
        if configured == "cuda":
            if not self._hardware.cuda_available:
                raise VLMUnsupportedConfigurationError("QWEN_VL_DEVICE=cuda but CUDA is unavailable.")
            return "cuda"
        if not self._hardware.cuda_available:
            return "cpu"
        free_mb = self._hardware.gpu_free_memory_mb or self._hardware.gpu_total_memory_mb or 0
        required_mb = int(self._minimum_cuda_gb(self._settings.qwen_vl_model_tier) * 1024)
        return "cuda" if free_mb >= required_mb else "cpu"

    def _resolve_dtype(self, device: str) -> str:
        configured = self._settings.qwen_vl_dtype
        if configured != "auto":
            if configured == "float16" and device == "cpu":
                raise VLMUnsupportedConfigurationError("float16 CPU inference is not supported by the Notify runtime policy.")
            return configured
        if device == "cpu":
            return "float32"
        # Prefer BF16 on GPUs that advertise support, otherwise FP16.
        try:
            import torch
            if torch.cuda.is_bf16_supported():
                return "bfloat16"
        except Exception:
            pass
        return "float16"

    def _validate_quantization(self, device: str) -> None:
        q = self._settings.qwen_vl_quantization
        if q != "none" and device != "cuda":
            raise VLMUnsupportedConfigurationError("4/8-bit VLM quantization is supported only on CUDA in this runtime.")
        if q != "none" and importlib.util.find_spec("bitsandbytes") is None and self._adapter_factory is None:
            raise VLMUnsupportedConfigurationError("bitsandbytes is required for configured VLM quantization.")

    def runtime_fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VLM_RUNTIME_ALGORITHM_VERSION,
                "model": self._resolved_model,
                "revision": self._settings.qwen_vl_model_revision,
                "device": self._resolved_device,
                "dtype": self._resolved_dtype,
                "quantization": self._settings.qwen_vl_quantization,
                "max_images": self._settings.qwen_vl_max_images_per_request,
                "max_edge": self._settings.qwen_vl_max_image_long_edge,
                "max_new_tokens": self._settings.qwen_vl_max_new_tokens,
            }
        )

    def _create_adapter(self, model_name: str):
        if self._adapter_factory is not None:
            return self._adapter_factory(model_name, self._resolved_device, self._resolved_dtype)
        from app.semantic_analysis.qwen_adapter import Qwen3VLAdapter
        return Qwen3VLAdapter(
            self._settings,
            model_name=model_name,
            device=self._resolved_device,
            dtype=self._resolved_dtype,
        )

    def _load_adapter(self) -> None:
        with self._load_lock:
            if self._adapter is not None and self._adapter.is_loaded():
                return
            adapter = self._create_adapter(self._resolved_model)
            try:
                adapter.load()
            except (MemoryError, VLMOutOfMemoryError) as exc:
                if not self._settings.qwen_vl_enable_fallback or self._resolved_model == self._fallback_model():
                    raise VLMOutOfMemoryError(str(exc)) from exc
                self._fallback_used = True
                self._fallback_reason = "RESOURCE_CONSTRAINT"
                self._resolved_model = self._fallback_model()
                adapter = self._create_adapter(self._resolved_model)
                adapter.load()
            self._adapter = adapter

    def is_loaded(self) -> bool:
        return bool(self._adapter and self._adapter.is_loaded())

    def inspect_runtime(self) -> VLMRuntimeMetadata:
        transformers_version = torch_version = None
        if self._adapter and self._adapter.is_loaded():
            transformers_version, torch_version = self._adapter.runtime_versions()
        return VLMRuntimeMetadata(
            configured_model_tier=self._settings.qwen_vl_model_tier,
            configured_model_name=self._settings.qwen_vl_model_name,
            primary_model=self._configured_primary_model(),
            resolved_model=self._resolved_model,
            fallback_model=self._fallback_model() if self._settings.qwen_vl_enable_fallback else None,
            fallback_enabled=self._settings.qwen_vl_enable_fallback,
            fallback_used=self._fallback_used,
            fallback_reason=self._fallback_reason,
            configured_device=self._settings.qwen_vl_device,
            resolved_device=self._resolved_device,
            resolved_dtype=self._resolved_dtype,
            quantization=self._settings.qwen_vl_quantization,
            model_revision=self._settings.qwen_vl_model_revision,
            model_loaded=self.is_loaded(),
            transformers_version=transformers_version,
            torch_version=torch_version,
            runtime_fingerprint=self.runtime_fingerprint(),
        )

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        if image.width <= 0 or image.height <= 0:
            raise VLMImageInputError("Invalid image dimensions.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        max_edge = self._settings.qwen_vl_max_image_long_edge
        longest = max(image.width, image.height)
        if longest > max_edge:
            scale = max_edge / longest
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
        return image

    def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        generation_config: VLGenerationConfig | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> VLMGenerationResult:
        if cancel_check and cancel_check():
            raise VLMCancelledError("Semantic inference cancelled before model load.")
        if not images or len(images) > self._settings.qwen_vl_max_images_per_request:
            raise VLMImageInputError("Image count is outside configured VLM bounds.")
        prepared = [self._prepare_image(image) for image in images]
        self._load_adapter()
        if cancel_check and cancel_check():
            raise VLMCancelledError("Semantic inference cancelled before generation.")
        config = generation_config or VLGenerationConfig(max_new_tokens=self._settings.qwen_vl_max_new_tokens)
        started = time.monotonic()
        with self._inference_semaphore:
            assert self._adapter is not None
            try:
                text, output_tokens = self._adapter.generate(prepared, prompt, config, cancel_check)
            except VLMOutOfMemoryError as exc:
                if not self._settings.qwen_vl_enable_fallback or self._resolved_model == self._fallback_model():
                    raise
                self._adapter.unload()
                self._fallback_used = True
                self._fallback_reason = "INFERENCE_OOM"
                self._resolved_model = self._fallback_model()
                self._adapter = self._create_adapter(self._resolved_model)
                self._adapter.load()
                text, output_tokens = self._adapter.generate(prepared, prompt, config, cancel_check)
        if cancel_check and cancel_check():
            raise VLMCancelledError("Semantic inference cancelled after generation.")
        return VLMGenerationResult(
            text=text,
            model_name=self._resolved_model,
            resolved_device=self._resolved_device,
            resolved_dtype=self._resolved_dtype,
            generation_seconds=time.monotonic() - started,
            input_image_count=len(prepared),
            output_token_count=output_tokens,
            fallback_used=self._fallback_used,
            fallback_reason=self._fallback_reason,
            runtime_fingerprint=self.runtime_fingerprint(),
        )

    async def generate_async(self, *args, **kwargs) -> VLMGenerationResult:
        return await asyncio.to_thread(self.generate, *args, **kwargs)

    def unload(self) -> None:
        with self._load_lock:
            if self._adapter is not None:
                self._adapter.unload()
            self._adapter = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
