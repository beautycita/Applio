import os

import requests


def hf_headers() -> dict:
    """Auth header for the private model repo, from HF_TOKEN or `hf auth login`."""
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_download(url: str, destination_path: str) -> None:
    """Download a Hub file to destination_path; fails loudly instead of saving an error page."""
    tmp_path = destination_path + ".part"
    with requests.get(url, headers=hf_headers(), stream=True, timeout=60) as response:
        if response.status_code in (401, 403, 404):
            raise RuntimeError(
                f"Could not download {url} (HTTP {response.status_code}). "
                "The model repo is private: run `hf auth login` with an account that can read it."
            )
        response.raise_for_status()
        with open(tmp_path, "wb") as file:
            for chunk in response.iter_content(1024 * 1024):
                file.write(chunk)
    os.replace(tmp_path, destination_path)
