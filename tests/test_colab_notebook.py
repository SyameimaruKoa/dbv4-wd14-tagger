import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "run_colab_server.ipynb"


def load_notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_colab_notebook_uses_dbv4_profile() -> None:
    source = load_notebook_source()

    assert 'MODEL_PROFILE = "balanced"' in source
    assert '"--model-profile", MODEL_PROFILE' in source
    for profile in (
        "lightweight",
        "balanced",
        "high",
        "ultra",
        "compact_manual",
        "medium_manual",
        "wd14_v3",
    ):
        assert f'"{profile}"' in source

    assert 'get_colab_secret("HF_TOKEN")' in source
    assert 'os.environ["HF_TOKEN"] = hf_token' in source


def test_colab_notebook_reads_tailscale_hostname_secret() -> None:
    source = load_notebook_source()

    assert 'get_colab_secret("TAILSCALE_HOSTNAME")' in source
    assert 'secret_hostname or (HOSTNAME or "google-colab").strip() or "google-colab"' in source
    assert 'f"--hostname={CONFIG_HOSTNAME}"' in source
