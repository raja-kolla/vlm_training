from cxr_vlm.data.prompts import build_assistant_response, build_user_prompt

__all__ = ["build_assistant_response", "build_user_prompt", "prepare_llava_dataset"]


def __getattr__(name: str):
    if name == "prepare_llava_dataset":
        from cxr_vlm.data.prepare_llava import prepare_llava_dataset
        return prepare_llava_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
