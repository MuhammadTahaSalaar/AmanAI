"""LLM module: model loading, prompt templates, fine-tuning, and inference."""

__all__ = ["ModelLoader", "PromptTemplates"]


def __getattr__(name: str):
    """Lazy imports so that ``transformers`` is not loaded at package-init time.

    This allows ``finetune.py`` (run via ``python -m src.llm.finetune``) to
    import unsloth *before* transformers, which is required for unsloth's
    kernel patches to take effect.
    """
    if name == "ModelLoader":
        from src.llm.model_loader import ModelLoader
        return ModelLoader
    if name == "PromptTemplates":
        from src.llm.prompt_templates import PromptTemplates
        return PromptTemplates
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
