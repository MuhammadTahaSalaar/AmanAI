"""LLM model loader for Llama 3.2 with GPU (4-bit NF4) or CPU (float32) support."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
# HF_TOKEN is loaded from .env by config.py via load_dotenv().
# huggingface_hub >= 0.17 reads HF_TOKEN from the environment automatically,
# so no explicit hf_login() call is needed (and it would require a network round-trip).


class ModelLoader:
    """Loads and manages the LLM for inference.

    On CUDA GPUs: loads the 4-bit NF4 quantized model via bitsandbytes.
    On CPU: loads a smaller 1B float32 model as a fallback.

    Supports optional LoRA adapter merging for fine-tuned checkpoints.
    """

    def __init__(
        self,
        model_name: str | None = None,
        lora_path: str | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self._requested_model = model_name or config.LLM_MODEL_NAME
        self._lora_path = lora_path or config.LORA_ADAPTER_PATH
        self._temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self._max_new_tokens = max_new_tokens or config.LLM_MAX_NEW_TOKENS

        self._model_name: str = self._requested_model
        self._cuda_available: bool = torch.cuda.is_available()
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Load the model and tokenizer into memory.

        Automatically selects GPU (4-bit quantized) or CPU (float32) mode
        based on hardware availability.
        """
        if self._cuda_available:
            self._load_gpu()
        else:
            self._load_cpu()

        if self._cuda_available and self._lora_path and Path(self._lora_path).exists():
            self._load_lora()
        elif self._lora_path and not self._cuda_available:
            logger.warning(
                "LoRA adapter skipped — adapter was trained for the GPU model "
                "and cannot be applied to the CPU fallback model.",
            )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Model loaded successfully [%s]", self._model_name)

    def _load_gpu(self) -> None:
        """Load 4-bit NF4 quantized model for GPU inference."""
        from transformers import BitsAndBytesConfig

        logger.info("CUDA detected — loading 4-bit NF4 model: %s", self._model_name)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16,
        )

    def _load_cpu(self) -> None:
        """Load float32 fallback model for CPU-only inference."""
        self._model_name = config.CPU_FALLBACK_MODEL
        logger.warning(
            "No CUDA GPU detected — falling back to CPU model: %s",
            self._model_name,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map="cpu",
            dtype=torch.float32,
        )

    def _load_lora(self) -> None:
        """Merge LoRA adapter weights into the base model."""
        from peft import PeftModel

        logger.info("Loading LoRA adapter from: %s", self._lora_path)
        self._model = PeftModel.from_pretrained(self._model, self._lora_path)
        self._model = self._model.merge_and_unload()
        logger.info("LoRA adapter merged successfully")

    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The full prompt string (system + context + query).

        Returns:
            The generated text response.
        """
        if self._model is None or self._tokenizer is None:
            self.load()

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the new tokens (skip the input prompt)
        input_length = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self._tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Strip complete role/control markers like <|report|>, <|enduser|>
        # and orphaned trailing fragments like "some text <|"
        import re as _re
        response = _re.sub(r"<\|[^|>]*\|>", "", response)   # complete <|...|>
        response = _re.sub(r"<\|[^|>]*$", "", response)      # orphaned trailing <|

        return response.strip()

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""
        return self._model is not None
