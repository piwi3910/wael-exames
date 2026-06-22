from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vlm_base_url: str = "http://192.168.10.246:8003/v1"
    vlm_model: str = "qwen3-vl"
    grader_base_url: str = "http://192.168.10.246:8888/v1"
    grader_model: str = "qwen3.6-35b"
    render_dpi: int = 200
    request_timeout: float = 180.0
    max_retries: int = 3
    out_dir: str = "out"


SETTINGS = Settings()
