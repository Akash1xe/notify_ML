from __future__ import annotations

import importlib
from typing import Sequence

from PIL import Image

from app.core.config import AppSettings
from app.core.exceptions import (
    VLMCancelledError,
    VLMDependencyMissingError,
    VLMInferenceError,
    VLMModelLoadError,
    VLMModelNotFoundError,
    VLMOutOfMemoryError,
)
from app.semantic_analysis.models import VLGenerationConfig


class _CancelStoppingCriteria:
    def __init__(self, check):
        self._check = check

    def __call__(self, input_ids, scores, **kwargs):
        return bool(self._check and self._check())


class Qwen3VLAdapter:
    def __init__(self, settings: AppSettings, *, model_name: str, device: str, dtype: str) -> None:
        self._settings = settings
        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._model = None
        self._processor = None
        self._torch = None
        self._transformers = None

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.is_loaded():
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise VLMDependencyMissingError(
                "Install Notify with the optional 'vlm' dependencies to use Qwen3-VL."
            ) from exc
        dtype = getattr(torch, self._dtype)
        kwargs = {
            "torch_dtype": dtype,
            "revision": self._settings.qwen_vl_model_revision,
            "cache_dir": str(self._settings.qwen_vl_model_cache_dir) if self._settings.qwen_vl_model_cache_dir else None,
            "local_files_only": self._settings.qwen_vl_local_files_only,
        }
        if self._settings.qwen_vl_quantization == "4bit":
            kwargs["load_in_4bit"] = True
        elif self._settings.qwen_vl_quantization == "8bit":
            kwargs["load_in_8bit"] = True
        try:
            processor = transformers.AutoProcessor.from_pretrained(
                self._model_name,
                revision=self._settings.qwen_vl_model_revision,
                cache_dir=kwargs["cache_dir"],
                local_files_only=self._settings.qwen_vl_local_files_only,
            )
            model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
            if model_cls is None:
                model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
            if model_cls is None:
                raise VLMDependencyMissingError("Installed Transformers does not support Qwen3-VL.")
            model = model_cls.from_pretrained(self._model_name, **kwargs)
            if self._device == "cuda" and not any(k.startswith("load_in_") for k in kwargs):
                model = model.to("cuda")
            self._processor, self._model = processor, model
            self._torch, self._transformers = torch, transformers
        except VLMDependencyMissingError:
            raise
        except OSError as exc:
            if self._settings.qwen_vl_local_files_only:
                raise VLMModelNotFoundError(f"Qwen model is not available in the local cache: {self._model_name}") from exc
            raise VLMModelLoadError(f"Unable to load Qwen model {self._model_name}") from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise VLMOutOfMemoryError(str(exc)) from exc
            raise VLMModelLoadError(str(exc)) from exc

    def generate(self, images: Sequence[Image.Image], prompt: str, generation_config: VLGenerationConfig, cancel_check=None):
        if not self.is_loaded():
            raise VLMModelLoadError("Qwen adapter is not loaded.")
        if cancel_check and cancel_check():
            raise VLMCancelledError("Semantic inference cancelled.")
        assert self._processor is not None and self._model is not None and self._torch is not None
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        try:
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=list(images), padding=True, return_tensors="pt")
            if self._device == "cuda":
                inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()}
            stopping_criteria = None
            if cancel_check and self._transformers is not None:
                stopping_criteria = self._transformers.StoppingCriteriaList([_CancelStoppingCriteria(cancel_check)])
            kwargs = {
                "max_new_tokens": generation_config.max_new_tokens,
                "do_sample": generation_config.do_sample,
                "repetition_penalty": generation_config.repetition_penalty,
            }
            if generation_config.do_sample:
                kwargs.update(temperature=generation_config.temperature, top_p=generation_config.top_p)
            if stopping_criteria is not None:
                kwargs["stopping_criteria"] = stopping_criteria
            with self._torch.inference_mode():
                generated = self._model.generate(**inputs, **kwargs)
            input_len = inputs["input_ids"].shape[1]
            output_ids = generated[:, input_len:]
            output = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return output, int(output_ids.shape[1])
        except VLMCancelledError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise VLMOutOfMemoryError(str(exc)) from exc
            raise VLMInferenceError(str(exc)) from exc
        except Exception as exc:
            raise VLMInferenceError(str(exc)) from exc

    def unload(self) -> None:
        self._model = None
        self._processor = None

    def runtime_versions(self):
        return (
            getattr(self._transformers, "__version__", None) if self._transformers else None,
            getattr(self._torch, "__version__", None) if self._torch else None,
        )
