from examgrader.config import SETTINGS

def test_settings_defaults_point_at_dgx():
    assert SETTINGS.vlm_base_url == "http://192.168.10.246:8003/v1"
    assert SETTINGS.vlm_model == "qwen3-vl"
    assert SETTINGS.grader_base_url == "http://192.168.10.246:8888/v1"
    assert SETTINGS.grader_model == "qwen3.6-35b"
    assert SETTINGS.render_dpi == 200
