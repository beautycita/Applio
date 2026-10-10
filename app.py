# Make sure the config file exists
import os
import shutil
import sys

# We need the CWD for finding the config file, but while we're at it, add it to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# TODO: This path is regenerated all over the place in ReMiFaSol
# should probably be in a static module for everything to reference
CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")

# The base config file to start from
CONFIG_TEMPLATE_PATH = os.path.join(now_dir, "assets", "config_template.json")

if not os.path.exists(CONFIG_PATH):
    print("Config file not found. Creating fresh from template.")
    shutil.copy(CONFIG_TEMPLATE_PATH, CONFIG_PATH)

# Plataform config
from rvc.lib.platform import platform_config

platform_config()

import argparse
import types
import gradio as gr
import pathlib
import logging

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

_ARG_PARSER = argparse.ArgumentParser(
    description="ReMiFaSol Web UI",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_ARG_PARSER.add_argument(
    "--port", type=int, default=DEFAULT_PORT, help="Server port (default: %(default)s)"
)
_ARG_PARSER.add_argument(
    "--server-name",
    type=str,
    default=DEFAULT_SERVER_NAME,
    help="Server hostname (default: %(default)s)",
)
_ARG_PARSER.add_argument(
    "--share", action="store_true", help="Create a public Gradio share link"
)
_ARG_PARSER.add_argument(
    "--open", action="store_true", help="Open the browser automatically"
)
_ARG_PARSER.add_argument(
    "--client", action="store_true", help="Enable client mode (mounts realtime API)"
)
_args, _ = _ARG_PARSER.parse_known_args()
client_mode = _args.client
_has_share = _args.share
_has_open = _args.open

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Fix Gradio NoneType error when entering an invalid value
gr.Number.preprocess = types.MethodType(
    lambda self, payload: (
        None
        if payload is None
        or (self.minimum is not None and payload < self.minimum)
        or (self.maximum is not None and payload > self.maximum)
        else self.round_to_precision(payload, self.precision)
    ),
    gr.Number,
)

# detect gradio
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

# AMD dilated-conv workaround
import rvc.lib.amd

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab
from tabs.realtime.realtime import realtime_tab
from tabs.tensorboard.tensorboard import tensorboard_tab

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=False,
)

# Initialize i18n
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Load theme
import assets.themes.loadThemes as loadThemes

ui_theme = loadThemes.load_theme() or "ParityError/Interstellar"

STUDIO_CSS = pathlib.Path(os.path.join(now_dir, "assets", "themes", "studio.css")).read_text()



def device_badge() -> str:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"{props.name} · {props.total_memory / 1024**3:.0f} GB"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple GPU (MPS)"
    return "CPU only"


# Define Gradio interface
with gr.Blocks(title="ReMiFaSol") as ReMiFaSol:
    gr.HTML(
        f"""
        <header class="rmfs-header">
          <div class="rmfs-brand">
            <span class="rmfs-notes">♩ ♪ ♫</span>
            <h1>ReMiFaSol</h1>
            <p>Voice conversion studio</p>
          </div>
          <nav>
            <span class="rmfs-badge" title="Device used for conversion and training">{device_badge()}</span>
            <a href="https://github.com/beautycita/ReMiFaSol" target="_blank">GitHub</a>
          </nav>
        </header>
        """
    )
    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Training")):
        train_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Realtime")):
        realtime_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("TensorBoard")):
        tensorboard_tab()

    with gr.Tab(i18n("Settings")):
        settings_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()



def launch_gradio(server_name: str, server_port: int) -> None:
    app, _, _ = ReMiFaSol.launch(
        favicon_path="assets/ICON.ico",
        share=_has_share,
        inbrowser=_has_open,
        server_name=server_name,
        server_port=server_port,
        # don't block yet: the proxy routes below must be registered first
        prevent_thread_lock=True,
        allowed_paths=["logs"],
        theme=ui_theme,
        css=STUDIO_CSS,
        js=(
            pathlib.Path(os.path.join(now_dir, "tabs", "realtime", "main.js")).read_text()
            if client_mode
            else None
        ),
    )

    # Mount TensorBoard proxy so it's accessible from any origin
    from rvc.lib.tools.launch_tensorboard import get_tb_url
    import httpx
    from fastapi import Request, Response

    @app.api_route(
        "/tensorboard/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    @app.api_route(
        "/tensorboard",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def tb_proxy(request: Request, path: str = ""):
        tb_url = get_tb_url()
        if not tb_url:
            return Response("TensorBoard not started", status_code=503)
        url = f"{tb_url.rstrip('/')}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url=url,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() not in ["host"]
                },
                content=await request.body(),
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    if client_mode:
        from rvc.realtime.client import app as fastapi_app

        app.mount("/api", fastapi_app)

    ReMiFaSol.block_thread()


if __name__ == "__main__":
    port = _args.port
    server = _args.server_name

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break
