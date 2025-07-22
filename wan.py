try:
    import os, argparse, gc, math, sys
    from pathlib import Path
    import platform
    import time
    import imageio
    import re
    import base64
    import cv2
    import numpy as np
    import html
    from io import BytesIO
    from accelerate import dispatch_model
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    import bitsandbytes as bnb
    import copy
    import torch
    from tqdm import tqdm
    from PIL import Image, ImageOps
    from huggingface_hub import snapshot_download
    from transformers import CLIPVisionModel
    from transformers import BitsAndBytesConfig
    from diffusers import WanImageToVideoPipeline, WanVACEPipeline, AutoencoderKLWan
    from diffusers.utils import load_image, export_to_video
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union
    import runpy
    import json
    import logging
    from html import escape
    import traceback
    import getpass
    import tempfile
    from bs4 import BeautifulSoup
    import hashlib, json, pathlib
    from types import MethodType
except ModuleNotFoundError as e:
    sys.stderr.write('Required module not found. Did you forget to activate the virtual environment?\n')
    sys.stderr.write(f'Original exception: {e}\n')
    sys.exit(1)
    

_init_done = False
def _bootstrap():
    global _init_done
    global IS_WINDOWS, SCRIPT_DIR, MODEL_DIR_I2V, MODEL_DIR_VACE
    global FILE_PREPEND, DONT_TORCH_COMPILE, MOUNT_POINT
    global REPO_ID_I2V, REPO_ID_VACE
    global oneTimeDebugStatement
    global _init_done
    global PYTHON_SCRIPT
    if _init_done:
        return                       # already patched / env-set
    _init_done = True
    
    IS_WINDOWS = platform.system() == "Windows"

    # Set temp dirs appropriately
    SCRIPT_DIR = Path(__file__).resolve().parent
    if IS_WINDOWS:
        print("Running on Windows")
        os.environ["TEMP"] = r"F:\Temp"
        os.environ["TMP"] = r"F:\Temp"
        MODEL_DIR_I2V = Path(r"G:\Wan2_1\wan_i2v")
        MODEL_DIR_VACE = Path(r"G:\Wan2_1\wan_vace")
        FILE_PREPEND = ""
        DONT_TORCH_COMPILE = True

        #We set the Tdr Delay to 10 from the default 2. Driver updates might restore the default, so warn the user if this ever happens
        try:
            import winreg
            key_path = r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ) as key:
                value, _ = winreg.QueryValueEx(key, "TdrDelay")
                if isinstance(value, int) and value >= 10:
                    pass  # Silent if value is good
                else:
                    print(f"WARNING: TdrDelay is set to {value}. Consider increasing it to 10 or more.")
        except Exception as e:
            print(f"Error reading TDR Delay: {e}")
    else:
        print("Running on Linux")
        #torch._inductor.config.triton.cudagraphs = False
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"        #Is this real? ChatGPT told me to set it due to OOM issues involving VRAM fragmentation
        torch._dynamo.config.recompile_limit = 256
        #print(torch._dynamo.config.accumulated_recompile_limit)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["TEMP"] = "/tmp"
        os.environ["TMP"] = "/tmp"
        DONT_TORCH_COMPILE = True

        username = getpass.getuser()
        MOUNT_POINT = Path(f"/media/{username}/Weights")
        if not os.path.ismount(MOUNT_POINT):
            sys.stderr.write(f'Error: Drive is not mounted at {MOUNT_POINT}\n')
            sys.exit(1)
        MODEL_DIR_I2V = MOUNT_POINT / "Wan2_1" / "wan_i2v"
        MODEL_DIR_VACE = MOUNT_POINT / "Wan2_1" / "wan_vace"
        FILE_PREPEND = SCRIPT_DIR

    REPO_ID_I2V = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    #REPO_ID_VACE = "Wan-AI/Wan2.1-VACE-14B"
    REPO_ID_VACE = "Wan-AI/Wan2.1-VACE-14B-Diffusers"

    oneTimeDebugStatement = True

    #This may have been one of those gpt-o3 hallucinations
    #SMT threads on Rocket Lake share resources; 8 fast cores often beat 16 logical threads for GEMM‑heavy transformer layers.
    #print(torch.__config__.show())
    torch.set_num_threads(8)            # physical cores
    torch.set_num_interop_threads(1)
    
    WanTransformer3DModel.forward              = custom_wan_forward
    WanTransformer3DModel.reset_teacache_state = reset_teacache_state
    WanTransformer3DModel.reset_magcache_state = reset_magcache_state
    WanTransformer3DModel.custom_wan_init      = custom_wan_init
    WanTransformer3DModel.custom_wan_onload    = custom_wan_onload
    WanTransformer3DModel.custom_wan_offload   = custom_wan_offload
    
    WanVACETransformer3DModel.forward              = custom_wan_vace_forward
    WanVACETransformer3DModel.reset_teacache_state = reset_teacache_state
    WanVACETransformer3DModel.reset_magcache_state = reset_magcache_state
    WanVACETransformer3DModel.custom_wan_init      = custom_wan_init
    WanVACETransformer3DModel.custom_wan_onload    = custom_wan_onload
    WanVACETransformer3DModel.custom_wan_offload   = custom_wan_offload
    WanVACETransformer3DModel.custom_wan_vace_onload  = custom_wan_vace_onload
    WanVACETransformer3DModel.custom_wan_vace_offload = custom_wan_vace_offload
    
    PYTHON_SCRIPT = ""
    with open(__file__, 'r') as file:
        PYTHON_SCRIPT = file.read()

                      
###############################################################################
# PUBLIC API
###############################################################################

def build_pipeline_i2v_public(*, use_sageattention=False):
    """
    Call this from a job script to obtain a ready-to-use pipeline.
    Equivalent to the old internal build_pipeline_i2v() but *idempotent*.
    """
    _bootstrap()
    return build_pipeline_i2v(use_sageattention=use_sageattention)

def build_pipeline_vace_public(*, use_sageattention=False):
    """
    Call this from a job script to obtain a ready-to-use pipeline.
    Equivalent to the old internal build_pipeline_vace() but *idempotent*.
    """
    _bootstrap()
    return build_pipeline_vace(use_sageattention=use_sageattention)

def prepare_VACE_inputs(
    frame_pairs = [],          # list of (frame_idx, img_path) or (frame_idx, kwargs) tuples
    references  = None,        # list of img_path or kwargs
    latent_overrides = [],     # list of (frame_idx, kwargs) tuples
    height      = 480,
    width       = 480,
    num_frames  = 81
):
    for i, (a, args) in enumerate(frame_pairs):
        if isinstance(args, str):
            image = load_image_from_file(args)
        elif isinstance(args, dict):
            image = load_image_from_file(**args)
        frame_pairs[i] = (a, resize_image_for_wan_480p(image, (height, width)))
        #debug_save_path = f"debug_input_frame_{a:03d}.png"
        #print(debug_save_path)
        #frame_pairs[i][1].save(debug_save_path)
        assert height == frame_pairs[0][1].height
        assert width  == frame_pairs[0][1].width
        
    for i, (a, args) in enumerate(latent_overrides):
        assert isinstance(args, dict)
        latent_overrides[i] = (a, load_latent_from_html(**args))
        
    if references is not None:
        for i, args in enumerate(references):
            if isinstance(args, str):
                image = load_image_from_file(args)
            elif isinstance(args, dict):
                image = load_image_from_file(**args)
            references[i] = resize_image_for_wan_480p(image, (height, width))
    
    video = [Image.new("RGB", (width, height), (128, 128, 128)) for _ in range(num_frames)]
    mask_white = Image.new("L", (width, height), 255)
    mask_black = Image.new("L", (width, height), 0)
    mask = [mask_white] * num_frames

    # ---------------------- 4. drop key-frames into their slots ---------------
    for idx, frame_img in frame_pairs:
        if not (0 <= idx < num_frames):
            raise IndexError(f"Frame index {idx} is out of bounds for num_frames={num_frames}. Remember frame indexes are zero-based!")
        video[idx] = frame_img
        mask[idx] = mask_black
        
    for idx, frame_img in latent_overrides:
        if not (0 <= idx < num_frames):
            raise IndexError(f"latent index {idx} is out of bounds for num_frames={num_frames}. Remember frame indexes are zero-based!")
        mask[idx] = mask_black
        
    return video, mask, references, latent_overrides
    
###############################################################################
# INSTALLER
###############################################################################

# ---------- helpers ----------
def download_repo(repoID, modelDir):
    print("Downloading model files…")
    snapshot_download(repo_id=repoID,
                      local_dir=str(modelDir),
                      resume_download=True,
                      max_workers=4)

def quantize_clip(modelDir):
    enc_path = modelDir / "image_encoder"
    marker   = enc_path / "quantized_bnb"
    if marker.exists():
        return
    print("Quantising CLIP vision encoder to 8‑bit…")
    enc = CLIPVisionModel.from_pretrained(
        enc_path, load_in_8bit=True, device_map="cpu")
    for f in enc_path.glob("*.safetensors"):
        f.unlink()
    enc.save_pretrained(enc_path)
    marker.touch()
    del enc; gc.collect()
                      
###############################################################################
# HELPER FUNCTIONS
###############################################################################

#  drop-in VRAM inspector for PyTorch, only required parameter is "tag"
#  want_names is useful for showing the actual variable names, but it does have a VRAM footprint
def vram_report(tag: str = "", *, top_k: int = 15,
                device: str = "cuda:0",
                want_names: bool = True,
                roots: tuple = ()):
    """
    Lightweight VRAM inspector.
      • shows allocator stats
      • lists `top_k` largest live CUDA tensors
      • if `want_names=True`, tries to attach a variable / attribute name
        (including tensor attributes hanging off objects in `roots`)
    Never keeps a tensor reference, so it’s OOM-safe.
    """
    import gc, sys, torch, re, inspect, types, collections

    dev = torch.device(device)
    MB = 1 << 20
    fmt = lambda b: f"{b / MB:9.2f} MB"

    # ------------------------------------------------------------------
    # 1. allocator stats
    # ------------------------------------------------------------------
    alloc   = torch.cuda.memory_allocated(dev)
    reserv  = torch.cuda.memory_reserved(dev)
    small   = torch.cuda.memory_stats(dev)['allocation.all.current']
    hdr     = f"[VRAM @ {tag}]" if tag else "[VRAM]"
    print(f"{hdr:<22} alloc={fmt(alloc)}  reserved={fmt(reserv)}"
          f"  (unalloc={fmt(reserv-alloc)})  smallChunks={small}")

    # ------------------------------------------------------------------
    # 2. optional – build  id(tensor) ➜ pretty-name  map
    # ------------------------------------------------------------------
    id2name = {}
    if want_names:
        # ---- 2a. locals / globals along call-stack --------------------
        frame = sys._getframe()
        try:
            while frame:
                for k, v in frame.f_locals.items():
                    if torch.is_tensor(v):
                        id2name[id(v)] = k
                frame = frame.f_back
        finally:
            del frame   # break ref-cycle immediately

        # ---- 2b. walk attributes under user-supplied roots -----------
        Q = collections.deque()
        for root in roots:
            Q.append(("", root))        #  ("prefix", obj)

        seen_obj = set()
        MAX_ATTR  = 10_000              # safety valve

        while Q and len(seen_obj) < MAX_ATTR:
            prefix, obj = Q.popleft()
            if id(obj) in seen_obj:
                continue
            seen_obj.add(id(obj))

            if torch.is_tensor(obj):
                id2name[id(obj)] = prefix.rstrip(".")
                continue

            # recurse one level for plain python objects / nn.Modules
            if isinstance(obj, (types.ModuleType, torch.nn.Module, object)):
                for name, val in inspect.getmembers(obj):
                    if name.startswith("__"):
                        continue
                    new_prefix = f"{prefix}{name}."
                    Q.append((new_prefix, val))

    # ------------------------------------------------------------------
    # 3. collect live CUDA tensors
    # ------------------------------------------------------------------
    entries = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.device == dev:
                sz = obj.numel() * obj.element_size()
                entries.append((sz,
                                id2name.get(id(obj), "tensor"),
                                str(obj.dtype),
                                tuple(obj.size())))
        except Exception:
            pass  # object disappeared mid-iteration

    # ------------------------------------------------------------------
    # 4. print largest-first
    # ------------------------------------------------------------------
    if not entries:
        print("    (no Python-visible CUDA tensors)")
        print("-" * 80)
        return

    for sz, name, dtype, shape in sorted(entries,
                                         key=lambda x: x[0],
                                         reverse=True)[:top_k]:
        print(f"    {name:<22} {fmt(sz)}  {dtype:<9}  {shape}")
    print("-" * 80)



def save_json(name, data, folder="calibration_curves"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    #print(f"* Saved {path}")

def fingerprint_from_string(key: str) -> str:
    """
    SHA-256-based 32-character fingerprint of an arbitrary key string.
    """
    return hashlib.sha256(key.encode()).hexdigest()[:32]

def get_last_frame_from_pil_list(pil_frames: List[Image.Image]) -> Image.Image:
    """
    Defensive wrapper – guarantees we always give the next stage a copy.
    """
    if not pil_frames:
        raise ValueError("Empty frame list – cannot chain.")
    return pil_frames[-1].copy()

def ensure_2d(arr):
    return [arr] if isinstance(arr, list) and not any(isinstance(sub, list) for sub in arr) else arr
    
def resize_image_for_wan(
    image: Image.Image,
    model: str = '480p',
    sizeOverride = None
) -> Image.Image:
    """
    Resize and letterbox a PIL image to a Wan 2.1–friendly size.

    Parameters:
      image (PIL.Image)     – input image
      model (str)           – '480p' or '720p'; selects the target sum limit

    Returns:
      PIL.Image             – resized (or unchanged) image, letterboxed if needed

    Notes:
      • For 480p, ensures (width + height) ≤ 640 + 640 = 1280  
      • For 720p, ensures (width + height) ≤ 960 + 960 = 1920  
      • Both dimensions are floored to a multiple of 16  
      • Pads with black to preserve aspect ratio
    """
    # pick the sum limit based on model
    image = image[-1] if isinstance(image, list) else image
    
    model = model.lower()
    if model == '480p':
        #sum_limit = 480 + 480
        sum_limit = 512 + 512
        #sum_limit = 640 + 640
    elif model == '720p':
        sum_limit = 720 + 720
        #sum_limit = 960 + 960
    else:
        raise ValueError("model must be '480p' or '720p'")
        
    if sizeOverride is not None:
        sum_limit = sizeOverride[0] + sizeOverride[1]

    orig_w, orig_h = image.size
    orig_sum = orig_w + orig_h

    BASE = 16

    # if already within limits and multiples of 16, just copy
    if orig_sum <= sum_limit and orig_w % BASE == 0 and orig_h % BASE == 0:
        return image.copy()

    # scale so that width + height == sum_limit
    scale = sum_limit / float(orig_sum)
    raw_w = orig_w * scale
    raw_h = orig_h * scale

    def floor_to_mult(x, base):
        return max(int(math.floor(x / base)) * base, base)

    target_w = floor_to_mult(raw_w, BASE)
    target_h = floor_to_mult(raw_h, BASE)

    # letterbox onto the (target_w, target_h) canvas
    return ImageOps.pad(
        image,
        (target_w, target_h),
        method=Image.LANCZOS,
        color=(0, 0, 0)
    )

# convenience wrappers if you like
def resize_image_for_wan_480p(image: Image.Image, sizeOverride=None) -> Image.Image:
    return resize_image_for_wan(image, model='480p', sizeOverride=sizeOverride)

def resize_image_for_wan_720p(image: Image.Image, sizeOverride=None) -> Image.Image:
    return resize_image_for_wan(image, model='720p', sizeOverride=sizeOverride)


def decode_frame_from_video_bytes(
    video_bytes,
    frameN=0,
    end_frame=-1,
    convert="RGB",
    remove_duplicates=False,
    diff_threshold=5.0
):
    """
    Decode video frames from the provided MP4 bytes.
    
    :param video_bytes: Binary MP4 data.
    :param frameN: If -1, returns a list of all frames in the video (possibly up to `end_frame`).
                   Otherwise, returns a single frame at index `frameN`.
    :param end_frame: If not -1, stop reading once this frame index is reached.
    :param convert: "RGB" or "RGBA" – final PIL Image mode.
    :param remove_duplicates: If True, tries to skip frames that are nearly identical
                             to the last kept frame.
    :param diff_threshold: Float controlling how big the pixel difference must be
                          before a frame is kept.
    :return: A single PIL image if frameN != -1, else a list of PIL images.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_path = temp_file.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        os.remove(temp_path)
        raise ValueError("Failed to open temporary video file with OpenCV.")

    try:
        if frameN == -1:
            frames = []
            frame_idx = 0
            last_kept_frame = None  # we'll store the last kept frame's numpy array here

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break  # No more frames

                if end_frame != -1 and frame_idx > end_frame:
                    break  # We passed the requested end frame

                # Convert from BGR (OpenCV) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Check if we should skip near-duplicates
                if remove_duplicates and last_kept_frame is not None:
                    # Calculate mean squared difference between this frame and the last kept frame
                    diff = np.mean((frame_rgb.astype(np.float32) - last_kept_frame) ** 2)
                    if diff < diff_threshold:
                        # This frame is "close enough" to the last one – skip it.
                        #frame_idx += 1
                        continue

                # If we're here, either duplicates are not being removed,
                # or the difference is big enough to keep the frame.
                pil_img = Image.fromarray(frame_rgb, mode="RGB")
                if convert.upper() == "RGBA":
                    pil_img = pil_img.convert("RGBA")

                frames.append(pil_img)
                last_kept_frame = frame_rgb.astype(np.float32)  # store numeric array for comparison
                frame_idx += 1

            if not frames:
                raise ValueError("No frames read from video.")

            return frames

        else:
            # Return a single frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameN)
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Failed to read frame {frameN} from video.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb, mode="RGB")
            if convert.upper() == "RGBA":
                pil_img = pil_img.convert("RGBA")

            return pil_img

    finally:
        cap.release()
        os.remove(temp_path)

def load_image_from_file(img_path, frameN=0, end_frame=-1, convert="RGB"):
    """
    Load an image or extract a frame(s) from a video file or HTML file containing embedded media.

    :param img_path: Path to the image, video, or HTML file.
    :param frameN: For videos: if -1, return all frames (as list), otherwise return frame at index.
    :param end_frame: For videos: optional frame index to stop processing at (if frameN == -1).
    :param convert: Target image mode ("RGB" or "RGBA").
    :return:
        - If img_path == "": returns a white image -> PIL.Image.Image
        - If path points to an image file (.png, .jpg, etc): returns -> PIL.Image.Image
        - If path points to a video (.mp4):
            - frameN == -1 -> list[PIL.Image.Image]
            - frameN != -1 -> PIL.Image.Image
        - If path is HTML:
            - with embedded video:
                - frameN == -1 -> list[PIL.Image.Image]
                - frameN != -1 -> PIL.Image.Image
            - with embedded base64 image only -> PIL.Image.Image
        - Unsupported extension -> raises ValueError
        - File not found -> raises FileNotFoundError
        - No valid embedded content -> raises ValueError
    """
    if img_path == "":
        print("Warning: img_path is empty, returning a default white image.")
        return Image.new(convert, (1024, 1024), "white")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Init image not found: {img_path}")
    ext = os.path.splitext(img_path.lower())[1]

    if ext in [".png", ".bmp", ".jpg", ".jpeg", ".webp", ".gif", ".tiff"]:
        return Image.open(img_path).convert(convert)
    elif ext == ".mp4":
        with open(img_path, "rb") as f:
            video_bytes = f.read()
        return decode_frame_from_video_bytes(video_bytes, frameN=frameN, end_frame=end_frame, convert=convert)
    elif ext == ".html":
        # Attempt to extract an embedded MP4 or image from the HTML
        with open(img_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # 1) Check for embedded MP4 first
        video_match = re.search(
            r'<source[^>]*src="data:video/mp4;base64,([^"]+)"',
            html_content
        )
        if video_match:
            # -- 1-a) if caller asked for a single frame, try lossless first --
            if frameN != -1:
                lossless_rx = (
                    rf'<img[^>]*id=["\']lossless_{frameN}["\'][^>]*'
                    r'src="data:image/[^;]+;base64,([^"]+)"'
                )
                lossless_match = re.search(lossless_rx, html_content, re.IGNORECASE)
                if lossless_match:
                    print("Found lossless at " + str(frameN))
                    img_b64 = lossless_match.group(1)
                    if convert == "base64":
                        return img_b64
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        pil_img = Image.open(BytesIO(img_bytes)).convert(convert)
                        return pil_img
                    except Exception as e:
                        raise ValueError(f"Error decoding lossless PNG: {e}")

            # -- 1-b) fall back to extracting from the embedded MP4 -----------
            base64_data = video_match.group(1)
            if convert == "base64":
                return base64_data
            else:
                try:
                    video_bytes = base64.b64decode(base64_data)
                except Exception as e:
                    raise ValueError(f"Error decoding base64 video data: {e}")
                try:
                    return decode_frame_from_video_bytes(video_bytes, frameN=frameN, end_frame=end_frame, convert=convert)
                except Exception as e:
                    raise ValueError(f"Error extracting frame from embedded video: {e}")
    
        # 2) If no video found, try to find the first embedded base64 image
        img_match = re.search(
            r'<img[^>]*src="data:image[^;]+;base64,([^"]+)"',
            html_content
        )
        if img_match:
            base64_data = img_match.group(1)
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                raise ValueError(f"Error decoding base64 image data: {e}")
            try:
                pil_image = Image.open(BytesIO(image_bytes)).convert(convert)
                return pil_image
            except Exception as e:
                raise ValueError(f"Error creating PIL Image from base64 data: {e}")
        
        raise ValueError("No embedded MP4 or base64 image found in the HTML file.")
        
    else:
        raise ValueError(f"Unsupported file type: {ext}")

LATENT_RE = re.compile(
    r"<div\s+id=[\"']latent_(\d+)[\"']\s+[^>]*>(.*?)<\/div>",
    re.IGNORECASE | re.DOTALL,
)
def load_latent_from_html(
    html_path: str | Path,
    frameN: int,
    map_location: str = "cpu",
) -> Optional[torch.Tensor]:
    """
    Parameters
    ----------
    html_path : str | Path
        Path to the HTML file that contains the embedded latent <div>s.
    frameN : int
        Frame number you want to retrieve (0-based).
    map_location : str
        torch.load() map_location argument; "cpu" by default.

    Returns
    -------
    torch.Tensor | None
        The latent tensor for the requested frame, or None
        if the file does not contain that frame.
    """
    html_text = Path(html_path).read_text(encoding="utf-8", errors="ignore")

    # Search for *all* latent divs once (fast enough for typical files)
    for match in LATENT_RE.finditer(html_text):
        idx = int(match.group(1))
        if idx != frameN:
            continue

        # Unescape HTML then base64-decode
        b64_text = html.unescape(match.group(2)).strip()
        raw_bytes = base64.b64decode(b64_text)

        # Load the tensor that torch.save() wrote
        tensor = torch.load(BytesIO(raw_bytes), map_location=map_location)
        return tensor

    # If we get here, no matching div was found
    return None
        
def ensure_folder_exists(filename):
    """Save the HTML content to a file."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Function to create a thumbnail from base64 data
def create_thumbnail(base64_data, max_width, max_height, img_format="WEBP"):
    # Decode the base64 data
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))

    # Resize the image while maintaining the aspect ratio
    image.thumbnail((max_width, max_height))
    width, height = image.size

    # Convert the image back to base64
    buffered = BytesIO()
    image.save(buffered, format=img_format)
    thumbnail_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    if img_format == "WEBP":
        htmlImageType="image/webp"
    elif img_format == "PNG":
        htmlImageType="image/png"

    return f"data:{htmlImageType};base64,{thumbnail_base64}", width, height
    
def mp4_to_base64(video_filepath):
    """Read MP4 file into memory, then delete the file."""
    # Read the file into memory
    with open(video_filepath, "rb") as f:
        mp4_data = f.read()

    # Cleanup the temp file
    os.remove(video_filepath)

    # Return Base64-encoded MP4
    return base64.b64encode(mp4_data).decode("utf-8")
    
def extract_frames(video_path):
    """
    Extract all frames from an MP4 and return them as a list of PIL Images.
    
    :param video_path: Path to the input video file.
    :return: List of PIL.Image objects in RGB mode.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV default) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        frames.append(pil_image)

    cap.release()
    return frames
   
def image_to_base64(image, lossless_mode = False, img_format="WEBP"):
    """Convert a PIL Image to a Base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=img_format, lossless=lossless_mode)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_video_html(videos_data, PYTHON_SCRIPT, job_script):
    """Generate an HTML string with embedded images, metadata, and the original Python script."""
    html_content = "<html><head><title>Generated Videos</title></head><body>"
    
    if not videos_data:
        raise ValueError("videos_data is empty")
        
    # ------------------------------------------------------------------
    # 1.  Decode all clips to temporary files
    # ------------------------------------------------------------------
    if len(videos_data) == 1:
        meta = videos_data[0]["metadata"]
        merged_b64 = videos_data[0]["video_base64"]
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            part_paths = []
            for idx, item in enumerate(videos_data):
                part_path = os.path.join(tmpdir, f"part_{idx}.mp4")
                with open(part_path, "wb") as f:
                    f.write(base64.b64decode(item["video_base64"]))
                part_paths.append(part_path)
                
            # metadata from the last clip
            meta        = videos_data[-1]["metadata"]
            fps         = meta["FPS"]
            vid_width   = meta["vid_width"]
            vid_height  = meta["vid_height"]
    
            out_path = os.path.join(tmpdir, "merged.mp4")
    
            # ------------------------------------------------------------------
            # 2.  Stream frames from every source into a single writer
            # ------------------------------------------------------------------
            writer = imageio.get_writer(
                out_path,
                format = "ffmpeg",
                codec  = "libx264",
                fps    = fps,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
    
            for p in part_paths:
                reader = imageio.get_reader(p, format="ffmpeg")
                for frame in reader:
                    if frame.shape[0:2] != (vid_height, vid_width):
                        # simple nearest-resize via Pillow to match target size
                        frame = np.asarray(
                            Image.fromarray(frame).resize(
                                (vid_width, vid_height), Image.NEAREST
                            )
                        )
                    writer.append_data(frame)
                reader.close()
    
            writer.close()
    
            with open(out_path, "rb") as f:
                merged_b64 = base64.b64encode(f.read()).decode("ascii")
        
    vid_id = 1
    img_id = 0
    #for data in reversed(videos_data):
    #    vid_id += 1
    #    base64_video     = data['video_base64']
    #    vid_width        = data['metadata']['vid_width']
    #    vid_height       = data['metadata']['vid_height']
    #    #strength         = data['metadata']['Strength']
    #    img_frames       = data['image_frames']
    #    prompt           = data['metadata']['Prompt']
    #    negative_prompt  = data['metadata']['Negative Prompt']
    #    seed             = data['metadata']['Seed']
    #    steps            = data['metadata']['Steps']
    #    guidance_scale   = data['metadata']['Guidance Scale']
    #    loras            = data['metadata']['LoRAs']
    #    model_path       = data['metadata']['Model']
    #    #motion_path      = data['metadata']['Motion Module']
    #    #controlnets      = data['metadata']['Controlnet Models']
    #    metadata_mode    = data['metadata']["Mode"]
    #    generatorStates  = data['metadata']['GeneratorStates']
    #    fps              = data['metadata']["FPS"]
    #    #mask_path        = data['metadata']['Mask']
    #    input_image_path = data['metadata']['InputImage1']
    
    # ------------------------------------------------------------------
    # 3.  Pull metadata from the LAST item
    # ------------------------------------------------------------------
    meta = videos_data[-1]["metadata"]
    base64_video     = merged_b64
    vid_width        = meta["vid_width"]
    vid_height       = meta["vid_height"]
    img_frames       = [frame for d in videos_data for frame in d['image_frames']]
    prompt           = meta["Prompt"]
    negative_prompt  = meta["Negative Prompt"]
    seed             = meta["Seed"]
    steps            = meta["Steps"]
    guidance_scale   = meta["Guidance Scale"]
    loras            = meta["LoRAs"]
    model_path       = meta["Model"]
    metadata_mode    = meta["Mode"]
    generatorStates  = meta["GeneratorStates"]
    fps              = meta["FPS"]
    input_image_path = meta['InputImage1']
    
    frame_latent_div = f"""<div id="frame_latents" style="display: none; justify-content: left; flex-wrap: wrap;">
    """
    frame_idx_list = []
    frame_offset = 0
    for idx, item in enumerate(videos_data):
        frame_latents = item["frame_latents"]
        for idy in frame_latents:
            buf = BytesIO()
            torch.save(frame_latents[idy].cpu(), buf)
            latent_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            frame_idx_list.append(idy+frame_offset)
            frame_latent_div += f"""<div id='latent_{idy+frame_offset}' style='display:none;'>{escape(latent_b64)}</div>
            """
        frame_offset += item["metadata"]['num_frames']
    frame_latent_div += f"""</div>
    """

    html_content += f"""
    <div style='margin-bottom: 20px;'>
        <div style="display: flex; align-items: center;">
            <a href="#" onclick="openImageInNewTab('generated_{vid_id}'); return false;">
                <video id="generated_video" width="{vid_width}" height="{vid_height}" controls autoplay loop muted>
                    <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
                </video>
            </a>
            <div style="display: flex; margin-left: 10px;">
                <button onclick="prevFrame('generated_video')" style="font-size: 32px; padding: 10px; border: none; margin: 5px; border-radius: 5px;" >
                    &lt;
                </button>
                <button onclick="nextFrame('generated_video')" style="font-size: 32px; padding: 10px; border: none; margin: 5px; border-radius: 5px;" >
                    &gt;
                </button>
            </div>
        </div>
        <br/>
        <button id="toggle_button2" onclick="toggleDiv('frames_div', 'flex')" style="font-size: small;">Show Frames</button>
        <div id="frames_div" style="display: none; justify-content: left; flex-wrap: wrap;">"""
    
    lossless_image_div = f"""<div id="lossless_images" style="display: none; justify-content: left; flex-wrap: wrap;">
    """    
    
    for image64 in img_frames:
        input_image_thumbnail_data, input_w, input_h = create_thumbnail(image64, vid_width / 4, vid_height / 4)
        style = ""
        if img_id in frame_idx_list:
            style = "border: 3px solid red;"
            lossless_data, input_w_, input_h_ = create_thumbnail(image64, vid_width, vid_height, "PNG")
            lossless_image_div += f"""
                <img id="lossless_{img_id}" style="display: none" src="{lossless_data}" alt="Frame {img_id}" title="Frame {img_id}" />"""
        html_content += f"""
            <img id="generated_{img_id}" style="{style}" src="{input_image_thumbnail_data}" alt="Frame {img_id}" title="Frame {img_id}" onclick="jumpToFrame(this)" />
        """
        img_id += 1
        
    html_content += f"""
        </div>
        """
    
    lossless_image_div += f"""
    </div>"""
    
    html_content += frame_latent_div
    html_content += lossless_image_div
    
    html_content += f"""<p><strong>Workflow:</strong><br/>"""
    input_image                                  = load_image_from_file(input_image_path)
    input_image_path                             = os.path.abspath(os.path.join(os.path.dirname(__file__), input_image_path)).replace("\\", "/")
    input_image_thumbnail_data, input_w, input_h = create_thumbnail(image_to_base64(input_image), *(x / 4 for x in input_image.size))
    html_content += f"""<a href="file:///{input_image_path}" data-width="{input_w}" data-height="{input_h}"><img src="{input_image_thumbnail_data}" alt="Thumbnail"></a>"""
    #if mode == "inpainting":
    #    mask                                  = load_image_from_file(mask_path)
    #    mask_path                             = os.path.abspath(os.path.join(os.path.dirname(__file__), mask_path)).replace("\\", "/")
    #    mask_thumbnail_data, thumb_w, thumb_h = create_thumbnail(image_to_base64(mask),        *(x / 4 for x in mask.size))
    #    html_content += f""" + <a href="file:///{mask_path}"        data-width="{thumb_w}" data-height="{thumb_h}"><img src="{mask_thumbnail_data}" alt="Thumbnail"></a></p>"""
        
    html_content += f"""
        <h3>Seed: {seed}</h3>
        <p><strong>Mode:</strong> {metadata_mode}</p>
        <p><strong>Prompt:</strong> {prompt}</p>
        <p><strong>Negative Prompt:</strong> {negative_prompt}</p>
        <p><strong>Steps:</strong> {steps}</p>
        <p><strong>Guidance Scale:</strong> {guidance_scale}</p>
        <p><strong>LoRAs:</strong> {loras}</p>
        <p><strong>Model Path:</strong> {model_path}</p>
        """
    
    html_content += f"""
    </div>
    <hr/>
    """
        
    states_json_escaped = escape(json.dumps(generatorStates))

    html_content += f"""
    <button id="toggle_button" onclick="toggleDiv('job_script', 'block')" style="font-size: small;">Show Job Script</button>
    <div id='job_script' style='display:none;'>
        <pre>{job_script}</pre>
    </div>
    <div id='python_script' style='display:none;'>
        <pre>{PYTHON_SCRIPT}</pre>
    </div>
    <div id='generator_states' style='display:none;'>{states_json_escaped}</div>
    <script>
        function openImageInNewTab(id) {{
            var imgSrc = document.getElementById(id).src;
            var newTab = window.open(imgSrc, "_self");
        }}
        function toggleDiv(divID, displayStyle) {{
            var scriptDiv = document.getElementById(divID);
            var btn = document.getElementById('toggle_button');
            if (scriptDiv.style.display === 'none') {{
                scriptDiv.style.display = displayStyle;
                btn.textContent = 'Hide Job Script';
            }} else {{
                scriptDiv.style.display = 'none';
                btn.textContent = 'Show Job Script';
            }}
        }}
        function jumpToFrame(imgElement) {{
            const fullId = imgElement.id;
            const frameString = fullId.replace('generated_', '');
            const frameNumber = parseInt(frameString, 10);
            const fps = {fps};
            const timeInSeconds = frameNumber / fps;
            const video = document.getElementById('generated_video');
            video.currentTime = timeInSeconds;
        }}
        function nextFrame(id) {{
            video = document.getElementById(id);
            video.pause();
            video.currentTime += (1 / {fps});
        }}
        function prevFrame(id) {{
            video = document.getElementById(id);
            video.pause();
            video.currentTime -= (1 / {fps});
        }}
    </script>
    """
    html_content += "</body></html>"
    return html_content

def save_html_file(html_content, filename):
    """Save the HTML content to a file."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(filename, "w") as f:
        f.write(html_content)

def generator_to_b64(gen: torch.Generator) -> str:
    """
    Convert the given torch.Generator's state to a Base64 string.
    """
    state_tensor = gen.get_state()                 # 1) Retrieve state as a torch.Tensor
    state_bytes = state_tensor.numpy().tobytes()   # 2) Convert to raw bytes
    state_b64 = base64.b64encode(state_bytes)      # 3) Base64-encode
    return state_b64.decode('utf-8')               # 4) Convert bytes->str for embedding in HTML

def b64_to_generator(state_b64: str) -> torch.Generator:
    """
    Create and return a new torch.Generator whose state
    is loaded from the given Base64 string.
    """
    state_bytes = base64.b64decode(state_b64)      # 1) Decode from Base64
    state_array = np.frombuffer(state_bytes, dtype=np.uint8)  # 2) Convert to NumPy array
    state_tensor = torch.from_numpy(state_array)              # 3) Convert NumPy array->torch.Tensor
    
    gen = torch.Generator()                        # 4) Create a new torch.Generator
    gen.set_state(state_tensor)                    # 5) Set its state
    return gen

def module_size_mb(module):
    total_params = sum(p.numel() * p.element_size() for p in module.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in module.buffers())
    total_bytes = total_params + total_buffers
    return total_bytes / 1024**2  # Convert to MB
       
###############################################################################
# COLOR FUNCTIONS
###############################################################################

def frames_to_lab(frames_bgr):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32) for f in frames_bgr]

def collect_mean_std(frames_lab):
    stack = np.concatenate([f.reshape(-1, 3) for f in frames_lab], axis=0)
    return stack.mean(0), stack.std(0)

def grade_clip(frames_bgr, mu_t, std_t, chroma_gain_max=1.2, chroma_gain_min=0.8):
    """
    Match means and stds, but clamp a/b scaling so colours
    never get blown out.  L channel is still fully matched.
    """
    out = []
    for f in frames_bgr:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        mu_s, std_s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)

        scale = std_t / std_s
        # clamp a & b channel gains
        scale[1:] = np.clip(scale[1:], chroma_gain_min, chroma_gain_max)

        lab = (lab - mu_s) * scale + mu_t
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        out.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return out
    
###############################################################################
# CORE FUNCTIONS
###############################################################################

def pin_module_tensors(module):
    for name, param in module.named_parameters(recurse=False):
        # Move to CPU if needed
        if param.device.type != 'cpu':
            param = param.detach().to('cpu', dtype=torch.float16)
        # Pin after confirming it's on CPU
        #pinned_param = param.pin_memory()
        pinned_param = param.detach().to(torch.float16).pin_memory()
        setattr(module, name, torch.nn.Parameter(pinned_param))

    for name, buffer in module.named_buffers(recurse=False):
        if buffer.device.type != 'cpu':
            buffer = buffer.detach().to('cpu', dtype=torch.float16)
        #pinned_buffer = buffer.pin_memory()
        pinned_buffer = buffer.detach().to(torch.float16).pin_memory()
        setattr(module, name, pinned_buffer)

    for child_name, child_module in module.named_children():
        pin_module_tensors(child_module)

    return module
    
def check_cuda_tensors(module):
    for name, param in module.named_parameters():
        if param.device.type != 'cpu':
            print(f"[Found CUDA param]: {name} on {param.device}")
    for name, buffer in module.named_buffers():
        if buffer.device.type != 'cpu':
            print(f"[Found CUDA buffer]: {name} on {buffer.device}")

def build_pipeline_i2v(*, use_sageattention=False):
    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_DIR_I2V / "image_encoder",
        torch_dtype=torch.float32,      # fp32 on CPU
        #load_in_8bit=True,             # convert in memory each run
        #device_map="auto"
        device_map=None
    ).to("cpu")
    
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_DIR_I2V,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to("cpu")
        
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_DIR_I2V,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        #low_cpu_mem_usage=False,
        #offload_state_dict=True,               #dont use these, these write alot of stuff to disk
        #offload_folder=OFFLOAD_DIR
    ) #.to("cpu")
    
    pipe.cuda_device = torch.device("cuda")
    
    #vae_size = module_size_mb(pipe.vae)
    #print(f"VAE total size: {vae_size:.2f} MB")

    print("Installing custom WanXFormersProcessor")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.use_sageattention = use_sageattention
    for blk in pipe.transformer.blocks:                   #This should be done after enable_xformers_memory_efficient_attention
        blk.attn1.processor = WanXFormersSelfAttnProcessor(use_sageattention=use_sageattention)      #Confirmed via print statements
        blk.attn2.processor = WanXFormersCrossAttnProcessor(use_sageattention=use_sageattention)
        #print(blk.attn1.add_k_proj)
        #print(blk.attn2.add_k_proj)
        assert blk.attn1.add_k_proj is None,     "blk.attn1.add_k_proj != None, code change required"
        assert blk.attn2.add_k_proj is not None, "blk.attn2.add_k_proj == None, code change required"
    pipe.transformer.custom_init_done = False
    
    #torch.backends.cuda.enable_mem_efficient_sdp(True)
    #pipe.enable_sequential_cpu_offload()   # GPU‑related
    #pipe.enable_model_cpu_offload()        # GPU‑related
    pipe.enable_attention_slicing()
    
    pipe.num_inference_steps = None
    pipe.model_name = REPO_ID_I2V
    pipe.prompt         = None
    pipe.negativePrompt = None
    
    return pipe

def build_pipeline_vace(*, use_sageattention=False):
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_DIR_VACE,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to("cpu")
        
    pipe = WanVACEPipeline.from_pretrained(
        MODEL_DIR_VACE,
        vae=vae,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        #low_cpu_mem_usage=False
    ) #.to("cpu")
    
    pipe.cuda_device = torch.device("cuda")
    
    #vae_size = module_size_mb(pipe.vae)
    #print(f"VAE total size: {vae_size:.2f} MB")

    print("Installing custom WanXFormersProcessor")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.use_sageattention = use_sageattention
    #Each block is 1340.46 MiB
    for blk in pipe.transformer.blocks:                   #This should be done after enable_xformers_memory_efficient_attention
        blk.attn1.processor = WanXFormersSelfAttnProcessor(use_sageattention=use_sageattention)      #Confirmed via print statements
        blk.attn2.processor = WanVACEXFormersCrossAttnProcessor(use_sageattention=use_sageattention)
        #print(blk.attn1.add_k_proj)
        #print(blk.attn2.add_k_proj)
        assert blk.attn1.add_k_proj is None, "blk.attn1.add_k_proj != None, code change required"
        assert blk.attn2.add_k_proj is None, "blk.attn2.add_k_proj != None, code change required"
    #Each VACE block is 1540.50 MiB
    for vace_blk in pipe.transformer.vace_blocks:
        vace_blk.attn1.processor = WanXFormersSelfAttnProcessor(use_sageattention=use_sageattention)
        vace_blk.attn2.processor = WanVACEXFormersCrossAttnProcessor(use_sageattention=use_sageattention)
        #print(vace_blk.attn1.add_k_proj)
        #print(vace_blk.attn2.add_k_proj)
        assert vace_blk.attn1.add_k_proj is None, "vace_blk.attn1.add_k_proj != None, code change required"
        assert vace_blk.attn2.add_k_proj is None, "vace_blk.attn2.add_k_proj != None, code change required"
    pipe.transformer.custom_init_done = False
    
    #torch.backends.cuda.enable_mem_efficient_sdp(True)
    #pipe.enable_sequential_cpu_offload()   # GPU‑related
    #pipe.enable_model_cpu_offload()        # GPU‑related
    pipe.enable_attention_slicing()
    
    pipe.num_inference_steps = None
    pipe.model_name = REPO_ID_VACE
    pipe.prompt         = None
    pipe.negativePrompt = None
    
    return pipe
    
from diffusers.models.attention_processor import XFormersAttnProcessor, Attention, AttnProcessor
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention

# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------
class _BaseWanXFormersProcessor:
    def __init__(self, attention_op=None, *, use_sageattention=False):
        self.attention_op = attention_op
        if use_sageattention:
            try:
                import sageattention            # pip install sageattention
            except ImportError as e:
                raise RuntimeError(
                    "use_sageattention=True but the package is not installed. "
                    "Run  `pip install sageattention`  first."
                ) from e
            
            def _identity_merge(t):
                return t  # Do nothing; retain (B,H,L,d)
            def _sage_xattn(q, k, v, scale, mask):
                return sageattention.sageattn(
                    q, k, v,
                    sm_scale=scale,
                    is_causal=False,
                    smooth_k=True,
                    tensor_layout="HND",
                )
            
            self._merge_heads = _identity_merge
            self._xattn       = _sage_xattn

    # ---- rotary helper (fp16-friendly, real math) -------------------------
    @staticmethod
    def _apply_rotary_fp16(t, freqs):
        """
        t: (B, H, L, d)    freqs: (1, 1, L, d/2) complex128 (cos + i·sin)
        returns same shape/dtype as t
        """
        dtype = t.dtype
        B, H, L, d = t.shape
        even, odd = t[..., ::2], t[..., 1::2]       # (B,H,L,d/2)

        if freqs.shape[1] == 1:                     # broadcast over heads
            freqs = freqs.expand(-1, H, -1, -1)

        cos, sin = freqs.real.to(dtype), freqs.imag.to(dtype)
        rot_even = even * cos - odd * sin
        rot_odd  = even * sin + odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2, -1)

    # ---- qkv projections, reshape, merge-heads ----------------------------
    @staticmethod
    def _project(attn, hidden_states, encoder_hidden_states):
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        return q, k, v

    @staticmethod
    def _split_heads(t, attn):
        return t.unflatten(2, (attn.heads, -1)).transpose(1, 2)  # (B,H,L,d)

    @staticmethod
    def _merge_heads(t):
        B, H, L, d = t.shape
        return t.reshape(B * H, L, d).contiguous()               # (B·H,L,d)

    # ---- xFormers fused attention ----------------------------------------
    def _xattn(self, q, k, v, scale, mask):
    #    return sage_attention(q, k, v, scale=scale, attn_mask=mask)
        return memory_efficient_attention(
            q, k, v,
            attn_bias=None if mask is None else mask,
            op=self.attention_op,
            scale=scale,
        )

    # ---- out projection ---------------------------------------------------
    @staticmethod
    def _proj_out(attn, x):
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)
        return x

# ---------------------------------------------------------------------------
#  Processor for attn1  (rotary always present)
# ---------------------------------------------------------------------------
class WanXFormersSelfAttnProcessor(_BaseWanXFormersProcessor):
    def __call__(self, attn: Attention, hidden_states, *, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb, **_):

        # -------- image-token split (Wan I2V) ------------------------------
        #enc_img = None
        #Commented out because add_k_proj is None
        #if attn.add_k_proj is not None:
        #    enc_img = encoder_hidden_states[:, :257]
        #    encoder_hidden_states = encoder_hidden_states[:, 257:]
        #if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

        # -------- QKV ------------------------------------------------------
        q, k, v = self._project(attn, hidden_states, encoder_hidden_states)
        q = self._split_heads(q, attn)
        k = self._split_heads(k, attn)
        v = self._split_heads(v, attn)

        # -------- rotary (always) -----------------------------------------
        q = self._apply_rotary_fp16(q, rotary_emb)
        k = self._apply_rotary_fp16(k, rotary_emb)

        # -------- fused attention -----------------------------------------
        qf, kf, vf = map(self._merge_heads, (q, k, v))
        out = self._xattn(qf, kf, vf, attn.scale, attention_mask)

        # -------- reshape back --------------------------------------------
        B, H, L, d = q.shape
        out = out.reshape(B, H, L, d).transpose(1, 2).flatten(2, 3)

        # -------- optional image branch -----------------------------------
        #if enc_img is not None:
        #    k_i = attn.norm_added_k(attn.add_k_proj(enc_img))
        #    v_i = attn.add_v_proj(enc_img)
        #    k_i = self._split_heads(k_i, attn)
        #    v_i = self._split_heads(v_i, attn)
        #    out += self._xattn(qf, self._merge_heads(k_i),
        #                       self._merge_heads(v_i), attn.scale, None
        #                       ).reshape(B, H, L, d).transpose(1, 2).flatten(2, 3)

        return self._proj_out(attn, out)

# ---------------------------------------------------------------------------
#  Processor for attn2  (no rotary)
# ---------------------------------------------------------------------------
class WanXFormersCrossAttnProcessor(_BaseWanXFormersProcessor):
    def __call__(self, attn: Attention, hidden_states, *, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, **_):

        # -------- image-token split ---------------------------------------
        #enc_img = None
        #if attn.add_k_proj is not None:
        enc_img = encoder_hidden_states[:, :257]
        encoder_hidden_states = encoder_hidden_states[:, 257:]
        #if encoder_hidden_states is None:
        #    encoder_hidden_states = hidden_states

        # -------- QKV ------------------------------------------------------
        q, k, v = self._project(attn, hidden_states, encoder_hidden_states)
        q = self._split_heads(q, attn)
        k = self._split_heads(k, attn)
        v = self._split_heads(v, attn)

        # (no rotary here)

        qf, kf, vf = map(self._merge_heads, (q, k, v))
        out = self._xattn(qf, kf, vf, attn.scale, attention_mask)

        B, H, L, d = q.shape
        out = out.reshape(B, H, L, d).transpose(1, 2).flatten(2, 3)

        #if enc_img is not None:
        k_i = attn.norm_added_k(attn.add_k_proj(enc_img))
        v_i = attn.add_v_proj(enc_img)
        k_i = self._split_heads(k_i, attn)
        v_i = self._split_heads(v_i, attn)
        out += self._xattn(qf, self._merge_heads(k_i),
                           self._merge_heads(v_i), attn.scale, None
                           ).reshape(B, H, L, d).transpose(1, 2).flatten(2, 3)

        return self._proj_out(attn, out)
    
# ---------------------------------------------------------------------------
#  Processor for attn2 VACE  (no rotary)
# ---------------------------------------------------------------------------
class WanVACEXFormersCrossAttnProcessor(_BaseWanXFormersProcessor):
    def __call__(self, attn: Attention, hidden_states, *, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, **_):

        # ---- QKV
        q, k, v = self._project(attn, hidden_states, encoder_hidden_states)
        q = self._split_heads(q, attn)
        k = self._split_heads(k, attn)
        v = self._split_heads(v, attn)

        # ---- single fused attention
        qf, kf, vf = map(self._merge_heads, (q, k, v))
        out = self._xattn(qf, kf, vf, attn.scale, attention_mask)

        # ---- merge heads & proj out
        B, H, L, d = q.shape
        out = out.reshape(B, H, L, d).transpose(1, 2).flatten(2, 3)
        return self._proj_out(attn, out)
    
from diffusers.models.transformers.transformer_wan_vace import (
    WanVACETransformer3DModel,
)
from diffusers.models.transformers.transformer_wan import (
    WanTransformer3DModel,
    Transformer2DModelOutput,
)
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
def custom_wan_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(getattr(self, "gpu_blocks", [])) != 2:
            raise RuntimeError(
                "gpu_blocks not initialised.  Call "
                "`transformer.custom_wan_onload(blocks=2, ...)` once BEFORE calibration."
            )
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states).to("cuda")              #dont cast the dtype, its a complex number
    
        self.patch_embedding = self.patch_embedding.to("cuda", dtype=torch.float16)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
    
        self.condition_embedder = self.condition_embedder.to("cuda", dtype=torch.float16)
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        
        torch.cuda.empty_cache()
        
        # ---------------------------------------------------------
        # Decide whether to compute the transformer this step
        # ---------------------------------------------------------
        #if True:
        #    do_compute = True
        #elif self.enable_magcache:
        if self.enable_magcache:
            if self.cnt == 0 or self.mag_prev_res is None:
                do_compute = True
            else:
                r_t        = self.mag_ratio_curve[self.cnt].item()
                est_e      = self.mag_err_acc + abs(r_t - 1.0)
                can_skip   = (est_e < self.mag_error_tol) and (self.mag_skip_cnt < self.mag_max_skip)
                do_compute = not can_skip
        elif self.enable_teacache:
            is_even = (self.cnt % 2 == 0)          # uncond vs cond
            mod_inp = timestep_proj if self.use_ref_steps else temb    # shape (1,6,dim) or (1,dim)
            if is_even:
                if self.prev_e0_even is None:
                    do_compute = True
                else:
                    rel_L1 = (mod_inp - self.prev_e0_even).abs().mean() / self.prev_e0_even.abs().mean()
                    self.acc_even += rel_L1.item()
                    if self.acc_even < self.teacache_threshold and self.cnt >= 2 and self.cnt < 900:
                        do_compute = False
                    else:
                        do_compute = True
                        self.acc_even = 0
                self.prev_e0_even = mod_inp.detach()
            else:  # odd
                if self.prev_e0_odd is None:
                    do_compute = True
                else:
                    rel_L1 = (mod_inp - self.prev_e0_odd).abs().mean() / self.prev_e0_odd.abs().mean()
                    self.acc_odd += rel_L1.item()
                    if self.acc_odd < self.teacache_threshold and self.cnt >= 2 and self.cnt < 900:
                        do_compute = False
                    else:
                        do_compute = True
                        self.acc_odd = 0
                self.prev_e0_odd = mod_inp.detach()
        else:
            do_compute = True
            
            
        if not do_compute:
            if self.enable_magcache:
                assert self.mag_prev_res is not None, "MagCache asked to skip but no residual saved"
                hidden_states     += self.mag_prev_res
                self.mag_skip_cnt += 1
                r_t                = self.mag_ratio_curve[self.cnt].item()
                self.mag_err_acc  += abs(r_t - 1.0)
            else:   # TeaCache
                if is_even:
                    assert self.prev_res_even is not None, "prev_res_even is None but computation was skipped"
                    hidden_states += self.prev_res_even
                else:
                    assert self.prev_res_odd is not None, "prev_res_odd is None but computation was skipped"
                    hidden_states += self.prev_res_odd
        else:
            if self.enable_magcache or self.enable_teacache:
                ori_x = hidden_states.clone()
                
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                print("WARNING! This codepath is not up to date and may fail in unknown ways!")
                #for block in self.blocks:
                for block in tqdm(self.blocks, desc="Transformer blocks (grad enabled)"):
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                    )
            else:
                #if self.custom_init_done != True:
                if len(self.gpu_blocks) != 2:
                    raise

                #I've commented this out as this is instantiated once on init instead
                #self.prefetch_stream = torch.cuda.Stream()
                for i, cpu_block in enumerate(self.blocks):
                #for i, cpu_block in tqdm(enumerate(self.blocks), total=len(self.blocks), desc="Transformer blocks"):
                    active   = self.gpu_blocks[i & 1]          # block used this iteration
                    prefetch = self.gpu_blocks[(i + 1) & 1]    # block we will fill for next iter

                    #A. this blocks until the previous copy is finished, if we didn't wait here then we might try to use the blocks before the copy completes
                    torch.cuda.current_stream().wait_stream(self.prefetch_stream)

                    #maybe fix bluescreen?
                    torch.cuda.synchronize()

                    #Fast in-place tensor copy  CPU->GPU   (src pinned on CPU, dest already on GPU).
                    if i+1 < len(self.blocks):
                        next_cpu_block = self.blocks[i + 1]
                    else:
                        next_cpu_block = self.blocks[0]
                    #B. It is at this point we've cleared all waits, and now we use the stream again to get the NEXT block, but not the one we're about to use
                    with torch.cuda.stream(self.prefetch_stream):
                        self.prefetch_stream.wait_event(self.done_lock)   # key line
                        for d_p, s_p in zip(prefetch.parameters(recurse=True), next_cpu_block.parameters(recurse=True)):
                            d_p.data.copy_(s_p.data.to(device="cuda", dtype=torch.float16, non_blocking=True))

                        for d_b, s_b in zip(prefetch.buffers(recurse=True), next_cpu_block.buffers(recurse=True)):
                            d_b.data.copy_(s_b.data.to(device="cuda", dtype=torch.float16, non_blocking=True))

                    # We split the active call out twice as doing it all in one go is too much VRAM used
                    # and it also causes a race condition (unconfirmed?)
                    #  B = 2 (cond / uncond) so we slice along dim-0
                    out_parts = []
                    for b in (0, 1):
                        hs  = hidden_states[         b : b+1 ]            # (1, ...)
                        enc = encoder_hidden_states[ b : b+1 ]
                        tpr = timestep_proj[         b : b+1 ]

                        out = active(hs, enc, tpr, rotary_emb)

                        if IS_WINDOWS:
                            out_parts.append(out)
                        else:
                            out_parts.append(out.clone())
                        #out_parts.append(out)

                    hidden_states = torch.cat(out_parts, dim=0)           # (2, ...)
                    del out_parts
                    torch.cuda.empty_cache()
                    #maybe fix bluescreen?
                    torch.cuda.synchronize()

                    # ---- record "my kernels queued" event ----
                    self.done_lock = torch.cuda.Event(enable_timing=False)
                    self.done_lock.record()                               # on stream-0
                    
            # --------- cache update after computation -----------
            if self.enable_magcache:
                residual = hidden_states - ori_x
                self.mag_prev_res = residual.detach()
                self.mag_skip_cnt = 0
                self.mag_err_acc  = 0.0
            elif self.enable_teacache:
                residual = hidden_states - ori_x
                if is_even:
                    self.prev_res_even = residual.detach()
                else:
                    self.prev_res_odd  = residual.detach()
        
        hidden_states         = hidden_states.to(        "cpu", dtype=torch.float32)
        temb                  = temb.to(                 "cpu", dtype=torch.float32)
        torch.cuda.empty_cache()

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
    
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
    
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
            
        if self.enable_magcache or self.enable_teacache:
            self.cnt = (self.cnt + 1) % 1000

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def custom_wan_vace_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: torch.Tensor = None,
    control_hidden_states_scale: torch.Tensor = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    #CUSTOM WAN CHANGE
    if len(getattr(self, "gpu_blocks", [])) != 0:
        raise RuntimeError(
            "gpu_blocks erroneously initialised. Dont call "
            "`transformer.custom_wan_onload()` outside this function."
        )
        
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    if control_hidden_states_scale is None:
        control_hidden_states_scale = control_hidden_states.new_ones(len(self.config.vace_layers))
    control_hidden_states_scale = torch.unbind(control_hidden_states_scale)
    if len(control_hidden_states_scale) != len(self.config.vace_layers):
        raise ValueError(
            f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} should be "
            f"equal to {len(self.config.vace_layers)}."
        )

    # 1. Rotary position embedding
    rotary_emb = self.rope(hidden_states).to("cuda")              #dont cast the dtype, its a complex number

    # 2. Patch embedding
    self.patch_embedding = self.patch_embedding.to("cuda", dtype=torch.float16)
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    self.vace_patch_embedding = self.vace_patch_embedding.to("cuda", dtype=torch.float16)
    control_hidden_states = self.vace_patch_embedding(control_hidden_states)
    control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)
    control_hidden_states_padding = control_hidden_states.new_zeros(
        batch_size, hidden_states.size(1) - control_hidden_states.size(1), control_hidden_states.size(2)
    )
    control_hidden_states = torch.cat([control_hidden_states, control_hidden_states_padding], dim=1)

    # 3. Time embedding
    self.condition_embedder = self.condition_embedder.to("cuda", dtype=torch.float16)
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    # 4. Image embedding
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Decide whether to compute the transformer this step
    # ---------------------------------------------------------
    if self.enable_magcache:
        if self.cnt == 0 or self.mag_prev_res is None:
            do_compute = True
        else:
            r_t        = self.mag_ratio_curve[self.cnt].item()
            est_e      = self.mag_err_acc + abs(r_t - 1.0)
            can_skip   = (est_e < self.mag_error_tol) and (self.mag_skip_cnt < self.mag_max_skip)
            do_compute = not can_skip
    elif self.enable_teacache:
        is_even = (self.cnt % 2 == 0)          # uncond vs cond
        mod_inp = timestep_proj if self.use_ref_steps else temb    # shape (1,6,dim) or (1,dim)
        if is_even:
            if self.prev_e0_even is None:
                do_compute = True
            else:
                rel_L1 = (mod_inp - self.prev_e0_even).abs().mean() / self.prev_e0_even.abs().mean()
                self.acc_even += rel_L1.item()
                if self.acc_even < self.teacache_threshold and self.cnt >= 2 and self.cnt < 900:
                    do_compute = False
                else:
                    do_compute = True
                    self.acc_even = 0
            self.prev_e0_even = mod_inp.detach()
        else:  # odd
            if self.prev_e0_odd is None:
                do_compute = True
            else:
                rel_L1 = (mod_inp - self.prev_e0_odd).abs().mean() / self.prev_e0_odd.abs().mean()
                self.acc_odd += rel_L1.item()
                if self.acc_odd < self.teacache_threshold and self.cnt >= 2 and self.cnt < 900:
                    do_compute = False
                else:
                    do_compute = True
                    self.acc_odd = 0
            self.prev_e0_odd = mod_inp.detach()
    else:
        do_compute = True

    # 5. Transformer blocks
    if not do_compute:
        if self.enable_magcache:
            assert self.mag_prev_res is not None, "MagCache asked to skip but no residual saved"
            hidden_states     += self.mag_prev_res
            self.mag_skip_cnt += 1
            r_t                = self.mag_ratio_curve[self.cnt].item()
            self.mag_err_acc  += abs(r_t - 1.0)
        else:   # TeaCache
            if is_even:
                assert self.prev_res_even is not None, "prev_res_even is None but computation was skipped"
                res = self.prev_res_even.to(hidden_states.device, hidden_states.dtype)
            else:
                assert self.prev_res_odd is not None, "prev_res_odd is None but computation was skipped"
                res = self.prev_res_odd.to(hidden_states.device, hidden_states.dtype)
            hidden_states += res
            del res
    else:
        if self.enable_magcache or self.enable_teacache:
            #ori_x = hidden_states.clone()
            #ori_x = ori_x.to("cpu")
            ori_x = hidden_states.to("cpu", copy=True)
            
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            print("WARNING! This codepath is not up to date and may fail in unknown ways!")
            #Code in this if block is unaltered
            # Prepare VACE hints
            control_hidden_states_list = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, control_hidden_states, timestep_proj, rotary_emb
                )
                control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[i]))
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                if i in self.config.vace_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale
        else:
            
            #vram_report(f"Before VACE blocks")
            # VACE blocks
            #If VRAM allows, we can speed this up by keeping both block 0 and block 1 in gpu to start
            self.custom_wan_vace_onload(use_sageattention=self.pipe.use_sageattention)
            control_hidden_states_list = []
            for i, cpu_block in enumerate(self.vace_blocks):
                active   = self.gpu_vace_blocks[0]          # block used this iteration
                prefetch = self.gpu_vace_blocks[0]
                
                cond_parts = []
                ctrl_parts = []
                for b in (0, 1):                                   # slice 0 / 1
                    hs  = hidden_states[         b : b + 1 ]
                    enc = encoder_hidden_states[ b : b + 1 ]
                    ctl = control_hidden_states[ b : b + 1 ]
                    tpr = timestep_proj[         b : b + 1 ]

                    conditioning_states_slice, control_hidden_states_slice = active(
                        hs, enc, ctl, tpr, rotary_emb
                    )
                    cond_parts.append(conditioning_states_slice)
                    ctrl_parts.append(control_hidden_states_slice)

                conditioning_states   = torch.cat(cond_parts,  dim=0)
                control_hidden_states = torch.cat(ctrl_parts, dim=0)
                control_hidden_states_list.append(
                    ( conditioning_states.to("cpu", dtype=torch.float16).contiguous(),
                      control_hidden_states_scale[i] ) #tuple
                )
                #control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[i]))
                del cond_parts, ctrl_parts, conditioning_states
                
                torch.cuda.empty_cache()
                #maybe fix bluescreen?
                torch.cuda.synchronize()
                
                #Because we only keep one block at a time, do the swap here
                if i == 0:    #vace block 0 has a different shape, so we must hard reset after using it
                    self.custom_wan_vace_offload()
                    self.custom_wan_vace_onload(start_block=1,use_sageattention=self.pipe.use_sageattention)
                elif i+1 < len(self.vace_blocks):
                    next_cpu_block = self.vace_blocks[i + 1]
                    for d_p, s_p in zip(prefetch.parameters(recurse=True), next_cpu_block.parameters(recurse=True)):
                        d_p.data.copy_(s_p.data.to(device="cuda", dtype=torch.float16, non_blocking=True))

                    for d_b, s_b in zip(prefetch.buffers(recurse=True), next_cpu_block.buffers(recurse=True)):
                        d_b.data.copy_(s_b.data.to(device="cuda", dtype=torch.float16, non_blocking=True))
            control_hidden_states_list = control_hidden_states_list[::-1]
            vace_layer_set = set(self.config.vace_layers)
            self.custom_wan_vace_offload()
            
            #vram_report(f"Before standard blocks")

            #Standard blocks
            self.custom_wan_onload(use_sageattention=self.pipe.use_sageattention)
            for i, cpu_block in enumerate(self.blocks):
                active   = self.gpu_blocks[i & 1]          # block used this iteration
                prefetch = self.gpu_blocks[(i + 1) & 1]    # block we will fill for next iter

                #Block until the previous copy is finished, if we didn't wait here then we might try to use the blocks before the copy completes
                torch.cuda.current_stream().wait_stream(self.prefetch_stream)

                #maybe fix bluescreen?
                torch.cuda.synchronize()

                #Fast in-place tensor copy  CPU->GPU   (src pinned on CPU, dest already on GPU).
                if i+1 < len(self.blocks):
                    next_cpu_block = self.blocks[i + 1]
                else:
                    next_cpu_block = self.blocks[0]
                with torch.cuda.stream(self.prefetch_stream):
                    self.prefetch_stream.wait_event(self.done_lock)   # key line
                    for d_p, s_p in zip(prefetch.parameters(recurse=True), next_cpu_block.parameters(recurse=True)):
                        d_p.data.copy_(s_p.data.to(device="cuda", dtype=torch.float16, non_blocking=True))

                    for d_b, s_b in zip(prefetch.buffers(recurse=True), next_cpu_block.buffers(recurse=True)):
                        d_b.data.copy_(s_b.data.to(device="cuda", dtype=torch.float16, non_blocking=True))
                
                out_parts = []
                #print(i)
                if i in vace_layer_set:
                    control_hint, scale = control_hidden_states_list.pop()
                    control_hint = control_hint.to("cuda", dtype=torch.float16)
                    scale        = scale.to(control_hint.dtype).to(control_hint.device)

                for b in (0, 1):
                    hs  = hidden_states[         b : b + 1 ]
                    enc = encoder_hidden_states[ b : b + 1 ]
                    tpr = timestep_proj[         b : b + 1 ]

                    hidden_states_slice = active(hs, enc, tpr, rotary_emb)

                    if i in vace_layer_set:
                        #control_hint, scale = control_hidden_states_list.pop()
                        control_hint_slice = control_hint[ b : b + 1 ]
                        hidden_states_slice = hidden_states_slice + control_hint_slice * scale

                    out_parts.append(hidden_states_slice)

                hidden_states = torch.cat(out_parts, dim=0)
                    
                del out_parts
                if i in vace_layer_set:
                    del control_hint, control_hint_slice
                torch.cuda.empty_cache()
                #maybe fix bluescreen?
                torch.cuda.synchronize()

                # ---- record "my kernels queued" event ----
                self.done_lock = torch.cuda.Event(enable_timing=False)
                self.done_lock.record()                               # on stream-0
            #vram_report(f"After Standard blocks")
            self.custom_wan_offload()
            #vram_report(f"After custom_wan_offload")

        # --------- cache update after computation -----------
        if self.enable_magcache:
            ori_x = ori_x.to(hidden_states.device, hidden_states.dtype)
            residual = hidden_states - ori_x
            self.mag_prev_res = residual.detach()
            self.mag_skip_cnt = 0
            self.mag_err_acc  = 0.0
            del ori_x
        elif self.enable_teacache:
            ori_x = ori_x.to(hidden_states.device, hidden_states.dtype)
            residual = hidden_states - ori_x                     # still on GPU
            cpu_residual = residual.to('cpu', dtype=torch.float16).contiguous()
            if is_even:
                self.prev_res_even = cpu_residual.detach()
            else:
                self.prev_res_odd  = cpu_residual.detach()
            del residual, ori_x
    
    hidden_states         = hidden_states.to(        "cpu", dtype=torch.float32)
    temb                  = temb.to(                 "cpu", dtype=torch.float32)
    torch.cuda.empty_cache()
    gc.collect()

    # 6. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if self.enable_magcache or self.enable_teacache:
        self.cnt = (self.cnt + 1) % 1000

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
        
def reset_teacache_state(self):
    self.cnt           = 0
    self.acc_even      = 0.0
    self.acc_odd       = 0.0
    self.prev_e0_even  = None
    self.prev_e0_odd   = None
    self.prev_res_even = 0
    self.prev_res_odd  = 0
    
def reset_magcache_state(self):
    self.cnt = 0
    self.mag_skip_cnt = 0
    self.mag_err_acc  = 0.0
    self.mag_prev_res = None
    
# This is called once ever
def custom_wan_init(
        self,
        pipe = None,
        teacache_threshold = 0.0,
        magcache_error_tol = 0.0,
        magcache_max_skip  = 99
):
    if self.custom_init_done:
        return
    
    self.pipe = pipe
    
    # ---------- Block Swap ----------
    #pin the transformer for more efficient memory swapping
    self.gpu_blocks = []
    for i, block in tqdm(enumerate(self.blocks), total=len(self.blocks), desc="Pinning transformer blocks"):
        check_cuda_tensors(block)  # Sanity: ensure clean start
        pinned_block = pin_module_tensors(block)
        self.blocks[i] = pinned_block  # Replace in-place
        del block
        torch.cuda.empty_cache()
        gc.collect()

    if isinstance(self, WanVACETransformer3DModel):
        self.gpu_vace_blocks = []
        for i, vace_block in tqdm(enumerate(self.vace_blocks), total=len(self.vace_blocks), desc="Pinning VACE transformer blocks"):
            check_cuda_tensors(vace_block)  # Sanity: ensure clean start
            pinned_vace_block = pin_module_tensors(vace_block)
            self.vace_blocks[i] = pinned_vace_block  # Replace in-place
            del vace_block
            torch.cuda.empty_cache()
            gc.collect()
            
    self.prefetch_stream = torch.cuda.Stream()
    self.done_lock = torch.cuda.Event(enable_timing=False) #This is a one time dummy lock that we create, but we will recreate it many times
    self.done_lock.record()                                # on stream-0
    
    # ---------- TeaCache ----------
    self.enable_teacache = teacache_threshold > 0.0
    if self.enable_teacache:
        self.teacache_threshold = teacache_threshold
        self.use_ref_steps      = True
        self.reset_teacache_state()
     
    # ---------- MagCache ----------
    self.enable_magcache = (magcache_error_tol > 0) and not self.enable_teacache
    if self.enable_magcache:
        self.mag_error_tol   = magcache_error_tol
        self.mag_max_skip    = magcache_max_skip
        print("mag_max_skip == " + str(self.mag_max_skip))
        self.reset_magcache_state()
        self.mag_ratio_curve = None
    
    self.custom_init_done = True

def custom_wan_make_gpu_worker(template, torch_compile=True, use_sageattention=False):
    with torch.inference_mode():
        blk = copy.deepcopy(template).to("cuda", dtype=torch.float16)
        #Unsure if its necessary to recreate the processors, but its probably needed since we're compiling them
        if DONT_TORCH_COMPILE or not torch_compile:
            return blk
        else:
            blk.attn1.processor = WanXFormersSelfAttnProcessor(use_sageattention=use_sageattention)
            blk.attn2.processor = WanXFormersCrossAttnProcessor(use_sageattention=use_sageattention)
            return torch.compile(blk,
                             mode="reduce-overhead",
                             dynamic=True, fullgraph=False)
        
def custom_wan_onload(self, *, blocks=2, torch_compile=True, use_sageattention=False):
    if blocks == 1:
        self.gpu_blocks = [
            custom_wan_make_gpu_worker(self.blocks[0], torch_compile, use_sageattention)
        ]
    else:
        self.gpu_blocks = [
            custom_wan_make_gpu_worker(self.blocks[0], torch_compile, use_sageattention),
            custom_wan_make_gpu_worker(self.blocks[1], torch_compile, use_sageattention)
        ]
    torch.cuda.synchronize()
def custom_wan_vace_onload(self, *, blocks=1, start_block=0, torch_compile=True, use_sageattention=False):
    if blocks == 1:
        self.gpu_vace_blocks = [
            custom_wan_make_gpu_worker(self.vace_blocks[start_block], torch_compile, use_sageattention)
        ]
    else:
        self.gpu_vace_blocks = [
            custom_wan_make_gpu_worker(self.vace_blocks[start_block    ], torch_compile, use_sageattention),
            custom_wan_make_gpu_worker(self.vace_blocks[start_block + 1], torch_compile, use_sageattention)
        ]
    torch.cuda.synchronize()

def custom_wan_offload(self):
    for blk in self.gpu_blocks:
        del blk
    self.gpu_blocks.clear()
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()      # best-effort clean-up of CUDA-graphs pools
    else:
        print("WARNING! ipc_collect not present")
def custom_wan_vace_offload(self):
    for blk in self.gpu_vace_blocks:
        del blk
    self.gpu_vace_blocks.clear()
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()      # best-effort clean-up of CUDA-graphs pools
    else:
        print("WARNING! ipc_collect not present")
    
        

import torch, math, psutil, gc

def _calib_fwd(
    self,
    hidden_states: torch.Tensor,         # (B,C,F,H,W)  – fp16 latents
    timestep:      torch.LongTensor,     # (B,)
    encoder_hidden_states: torch.Tensor, # (B,L,D_txt)
    encoder_hidden_states_image=None,    # (B,257,D_img) or None
    return_dict:   bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
):
    # ------------------------------------------------------------
    # 0.  Helpers / one-time allocations
    # ------------------------------------------------------------
    if not hasattr(self, "prefetch_stream"):
        self.prefetch_stream = torch.cuda.Stream()
    if not hasattr(self, "done_lock"):
        self.done_lock = torch.cuda.Event(enable_timing=False)
        self.done_lock.record()
    if len(getattr(self, "gpu_blocks", [])) != 2:
        raise RuntimeError(
            "gpu_blocks not initialised.  Call "
            "`transformer.custom_wan_onload(blocks=1, ...)` once BEFORE calibration."
        )

    # ------------------------------------------------------------
    # 2.  Standard Wan plumbing
    # ------------------------------------------------------------
    
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w  = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states).to("cuda")                    # complex64 (device-correct)

    self.patch_embedding = self.patch_embedding.to("cuda", dtype=torch.float16)
    hidden_states = self.patch_embedding(hidden_states)                  # (B,D,F',H',W')
    hidden_states = hidden_states.flatten(2).transpose(1, 2)             # (B,N,D)
    self.patch_embedding = self.patch_embedding.to("cpu")    # free ~0.5 GB
    torch.cuda.empty_cache()

    self.condition_embedder = self.condition_embedder.to("cuda", dtype=torch.float16)
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        
    # ------------------------------------------------------------------
    # 3.  Text / image context  ➜  *project* to inner_dim (5120)
    # ------------------------------------------------------------------
        
    # off-load embedder
    self.condition_embedder = self.condition_embedder.to("cpu")
        
    #torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 4.  Block-swap loop
    # ------------------------------------------------------------
    hidden_states = hidden_states.to("cpu")
    torch.cuda.empty_cache()
    ori_x = hidden_states
    for i, cpu_block in enumerate(self.blocks):
        active   = self.gpu_blocks[i & 1]
        prefetch = self.gpu_blocks[(i + 1) & 1]

        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        
        torch.cuda.synchronize()

        # schedule next copy
        if i+1 < len(self.blocks):
            next_cpu_block = self.blocks[i + 1]
        else:
            next_cpu_block = self.blocks[0]
        with torch.cuda.stream(self.prefetch_stream):
            torch.cuda.current_stream().wait_event(self.done_lock)
            for d_p, s_p in zip(prefetch.parameters(recurse=True), next_cpu_block.parameters(recurse=True)):
                d_p.data.copy_(s_p.data.to("cuda", dtype=torch.float16, non_blocking=True))
            for d_b, s_b in zip(prefetch.buffers(recurse=True), next_cpu_block.buffers(recurse=True)):
                d_b.data.copy_(s_b.data.to("cuda", dtype=torch.float16, non_blocking=True))
            
        out_parts = []
        for b in range(batch_size):                          # B = 2
            hs  = hidden_states[ b : b+1 ].to("cuda", dtype=torch.float16)
            enc = encoder_hidden_states[ b : b+1 ]
            tpr = timestep_proj[ b : b+1 ]   #.to("cuda", dtype=torch.float16)

            torch.cuda.empty_cache()
            out  = active(hs, enc, tpr, rotary_emb)
            
            out_parts.append(out.to("cpu", dtype=torch.float16))             # immediately back to CPU
            del hs, enc, tpr, out                   # release GPU memory
            #del out
            torch.cuda.empty_cache()
        hidden_states = torch.cat(out_parts, dim=0)
        del out_parts
        torch.cuda.empty_cache()
        
        self.done_lock = torch.cuda.Event(enable_timing=False)
        self.done_lock.record()

    torch.cuda.empty_cache()
    #ori_x = ori_x.to("cuda")
    hidden_states = hidden_states.to("cuda")
    ori_x = ori_x.to("cuda")
    residual = hidden_states - ori_x
    residual = residual.to("cuda")

    # ------------------------------------------------------------
    # 5.  MagCache stats (still fp32 for numeric robustness)
    # ------------------------------------------------------------
    res32  = residual.float()
    prev32 = self.residual_cache[self.cnt & 1]
    if self.cnt >= 2 and prev32 is not None:
        # --- move ONLY for the norm calculation ---
        prev32_cuda = prev32.to(res32.device, dtype=res32.dtype)

        ratio = (res32.norm(dim=-1) / prev32_cuda.norm(dim=-1)).mean().item()
        std   = (res32.norm(dim=-1) / prev32_cuda.norm(dim=-1)).std ().item()
        cos   = (1 - F.cosine_similarity(res32, prev32_cuda,
                                         dim=-1, eps=1e-8)).mean().item()

        self.norm_ratio.append(round(ratio, 5))
        self.norm_std  .append(round(std,   5))
        self.cos_dis   .append(round(cos,   5))

        # free the temporary copy
        del prev32_cuda
        torch.cuda.empty_cache()

    # keep *detached* fp16 residual for future step, but on **CPU**
    #self.residual_cache[self.cnt & 1] = res32.detach()
    self.residual_cache[self.cnt & 1] = residual.detach().to("cpu", torch.float16)
    del res32, prev32
    torch.cuda.empty_cache()          # ~812 MiB returns immediately

    self.cnt += 1
    if self.cnt >= self.num_steps:
        save_json("magcache_ratio", self.norm_ratio)
        save_json("magcache_std",   self.norm_std)
        save_json("magcache_cos",   self.cos_dis)
        self.cnt = 0
        #self.norm_ratio.clear();      #needed to create the magcache file, dont clear it
        self.norm_std.clear();
        self.cos_dis.clear()

    # ------------------------------------------------------------
    # 6.  Tail – fp16
    # ------------------------------------------------------------
    self.scale_shift_table.data = self.scale_shift_table.data.to(hidden_states.device, dtype=torch.float16)
    self.norm_out = self.norm_out.to(hidden_states.device, dtype=torch.float16)
    self.proj_out = self.proj_out.to(hidden_states.device, dtype=torch.float16)
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
        
    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    
    # aggressively delete everything big
    self.scale_shift_table.data = self.scale_shift_table.data.to("cpu")
    hidden_states_cpu = hidden_states.to("cpu", dtype=torch.float16)
    del hidden_states
    del shift, scale, temb
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    from diffusers.utils import BaseOutput
    class _Dummy(BaseOutput):
        sample: torch.FloatTensor
    return _Dummy(sample=hidden_states_cpu)

#VACE not implemented yet, need to change the transformer call
def build_magcache_if_needed(
        pipe,
        latents,
        condition,
        timesteps,
        prompt_embeds,
        negative_prompt_embeds,
        image_embeds,
        *,
        sample_steps,
        cache_dir,
        fingerprint,
        transformer_dtype,
        attention_kwargs,
        guidance_scale
):
    """Calibrates MagCache *in-place* on the live latents if the curve file is
    missing. Returns nothing; afterwards pipe.transformer.mag_ratio_curve is ready."""
    import json, pathlib, torch, copy, math, types
    
    pipe.transformer.custom_wan_onload(blocks=2, torch_compile=True, use_sageattention=pipe.use_sageattention)
    
    pipe.transformer.reset_magcache_state()

    curve_file = pathlib.Path(cache_dir) / f"{fingerprint}.json"
    if curve_file.exists():
        pipe.transformer.mag_ratio_curve = torch.tensor(
            json.loads(curve_file.read_text()), dtype=torch.float32
        )
        return latents, condition                 # be sure to return latents and condition

    tr = pipe.transformer

    # ---- 1. snapshot -----------------------------------------------------------------
    lat0        = latents.clone()                        # (keep on CPU)
    cond0       = condition.clone()
    proj_out_device = next(tr.proj_out.parameters()).device
    proj_out_dtype = next(tr.proj_out.parameters()).dtype
    #gen_states  = [g.get_state() for g in torch.random.get_rng_state()]  # optional

    # ---- 2. run once with skipping disabled -------------------------------------------
    tr.enable_magcache = False          # skip logic OFF
    tr.forward         = types.MethodType(_calib_fwd, tr)   # collect norms
    tr.cnt          = 0
    tr.num_steps    = len(timesteps)
    tr.norm_ratio   = []
    tr.norm_std     = []
    tr.cos_dis      = []
    tr.residual_cache = [None, None]

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Calibrating..."):
    #for i, t in enumerate(timesteps):
        #latent_in  = torch.cat([latents, condition], dim=1).to(transformer_dtype)
        #t_in       = t.expand(latents.shape[0])

        #latent_in  = latent_in .to("cuda", torch.float16)
        #img_in     = image_embeds          .to("cuda", torch.float16)
        #txt_in     = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to("cuda", torch.float16)

        #noise_pair = tr(hidden_states=latent_in,
        #                timestep=t_in.to(torch.float16).to("cuda"),
        #                encoder_hidden_states=txt_in,
        #                encoder_hidden_states_image=img_in,
        #                #attention_kwargs=attention_kwargs,
        #                return_dict=False)[0]
        
        pipe._current_timestep = t
        latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
        timestep = t.expand(latents.shape[0])

        latent_model_input     = latent_model_input.to("cuda", torch.float16)
        image_embeds           = image_embeds.to("cuda", torch.float16)
        negative_prompt_embeds = negative_prompt_embeds.to("cuda", torch.float16)
        prompt_embeds          = prompt_embeds.to("cuda", torch.float16)
        timestep               = timestep.to("cuda", torch.float16)
        
        # -- 1. build a virtual 2× batch --------------------------------
        latent_in  = latent_model_input.expand(2, *latent_model_input.shape[1:])
        img_embeds = image_embeds.expand(2, *image_embeds.shape[1:])
        txt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        timestep_in = timestep.expand(2)          # shape (2,)
        
        noise_pair = pipe.transformer(
                hidden_states              = latent_in,
                timestep                   = timestep_in,
                encoder_hidden_states      = txt_embeds,
                encoder_hidden_states_image= img_embeds,
                return_dict=False,
            )[0]

        noise_uncond, noise_pred = noise_pair.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    #curve = torch.tensor(tr.norm_ratio, dtype=torch.float32)
    curve = [1.0, 1.0] + tr.norm_ratio          # prepend dummies
    curve = torch.tensor(curve, dtype=torch.float32)
    curve_file.parent.mkdir(parents=True, exist_ok=True)
    #curve_file.write_text(json.dumps(tr.norm_ratio))
    curve_file.write_text(json.dumps(curve.tolist()))
    pipe.transformer.mag_ratio_curve = curve            # ready for pass-2
    tr.norm_ratio.clear()

    # ---- 3. restore snapshot -----------------------------------------------------------
    latents   = lat0
    condition = cond0
    pipe.scheduler.set_timesteps(sample_steps, device=latents.device)
    # restore RNG if you rely on dropout / stochastic layers
    #for s, g in zip(gen_states, torch.random._get_rng_state()):
    #    g.set_state(s.cpu())

    # ---- 4. restore normal forward + enable skipping ----------------------------------
    from types import MethodType
    tr.forward = MethodType(custom_wan_forward, tr)
    tr.enable_magcache = True
    
    tr.proj_out.to(proj_out_device, dtype=proj_out_dtype)
    
    pipe.transformer.custom_wan_offload()

    return latents, condition       # caller replaces its variables
    

from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
def wan_i2v_custom_pipeline(
    pipe,
    image: PipelineImageInput,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    seed = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    image_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    save_lossless_frames: Optional[List[int]] = None,   # <-- NEW
):
    with torch.inference_mode():
        if generator == None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
    
        # 1. Check inputs. Raise error if not correct
        pipe.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
        )
    
        if num_frames % pipe.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {pipe.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)
    
        pipe._guidance_scale = guidance_scale
        pipe._attention_kwargs = attention_kwargs
        pipe._current_timestep = None
        pipe._interrupt = False
    
        device = pipe._execution_device
    
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
    
        # 3. Encode input prompt
        if pipe.prompt != prompt or pipe.negativePrompt != negative_prompt:
            pipe.prompt = prompt
            pipe.negativePrompt = negative_prompt
            pipe.set_progress_bar_config(desc="Pre-encoding text prompt")
            with pipe.progress_bar(total=1) as progress_bar:
                pipe.prompt_embeds, pipe.negative_prompt_embeds = pipe.encode_prompt(
                    prompt                      = prompt,
                    negative_prompt             = negative_prompt,
                    #do_classifier_free_guidance = pipe.do_classifier_free_guidance,
                    do_classifier_free_guidance = True,                      #throws an exception if you try to access the member variable for some reason
                    num_videos_per_prompt       = num_videos_per_prompt,
                    prompt_embeds               = None,
                    negative_prompt_embeds      = None,
                    max_sequence_length         = max_sequence_length,
                    device                      = "cpu",
                )
                progress_bar.update()
        else:
            pipe.prompt_embeds, pipe.negative_prompt_embeds = pipe.encode_prompt(
                prompt                      = prompt,
                negative_prompt             = negative_prompt,
                do_classifier_free_guidance = pipe.do_classifier_free_guidance,
                num_videos_per_prompt       = num_videos_per_prompt,
                prompt_embeds               = pipe.prompt_embeds,
                negative_prompt_embeds      = pipe.negative_prompt_embeds,
                max_sequence_length         = max_sequence_length,
                device                      = device,
            )
    
        # Encode image embedding
        transformer_dtype = pipe.transformer.dtype
        pipe.prompt_embeds = pipe.prompt_embeds.to(transformer_dtype)
        if pipe.negative_prompt_embeds is not None:
            pipe.negative_prompt_embeds = pipe.negative_prompt_embeds.to(transformer_dtype)
        
        if image_embeds is None:
            image_embeds = pipe.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)
    
        # 4. Prepare timesteps
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
    
        # 5. Prepare latent variables
        
        #print(f"pipe vae dtype == {pipe.vae.dtype}")
        pipe.set_progress_bar_config(desc="Prepare latent variables")
        pipe.vae.to(pipe.cuda_device, dtype=torch.float32)
        with pipe.progress_bar(total=1) as progress_bar:
            num_channels_latents = pipe.vae.config.z_dim
            image = pipe.video_processor.preprocess(image, height=height, width=width).to(pipe.vae.device, dtype=torch.float32)
            latents, condition = pipe.prepare_latents(
                image,
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                #dtype     = torch.float32,
                dtype     = pipe.vae.dtype,
                device    = pipe.vae.device,
                generator = generator,
                latents   = latents,
            )
            pipe.vae.to("cpu", dtype=torch.float32)
            latents = latents.to("cpu", dtype=torch.float32)
            condition = condition.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
            progress_bar.update()
        
    
        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        pipe._num_timesteps = len(timesteps)
    
        #print(pipe.transformer)
                
        pipe.set_progress_bar_config(desc="", mininterval=0.0, miniters=1)
        #if pipe.do_classifier_free_guidance:
        #    progress_scale = 3
        #else:
        #    progress_scale = 2
        
        if pipe.transformer.enable_magcache and pipe.transformer.mag_ratio_curve is None:
            fingerprint = fingerprint_from_string(pipe.model_name + pipe.embedding_string + pipe.lora_string + "|" + str(num_inference_steps) + "|" + str(num_frames) + "|" + str(height) + "x" + str(width))
            if pipe.num_inference_steps is not None:
                assert pipe.num_inference_steps == num_inference_steps, "magcache isn't calibrated for variable sample_steps"
                pipe.num_inference_steps = num_inference_steps
            latents, condition = build_magcache_if_needed(
                pipe,
                latents, condition,
                timesteps,
                pipe.prompt_embeds, pipe.negative_prompt_embeds, image_embeds,
                sample_steps=num_inference_steps,
                cache_dir="magcache_json",
                fingerprint=fingerprint,
                transformer_dtype=transformer_dtype,
                attention_kwargs=attention_kwargs,
                guidance_scale=guidance_scale
            )
        
        #make sure we have our prefetch gpu_block ready
        pipe.transformer.custom_wan_onload(use_sageattention=pipe.use_sageattention)
    
        if pipe.transformer.enable_magcache:
            pipe.transformer.reset_magcache_state()
        
        #Do not worry, the largest of these is 6 MB
        image_embeds           = image_embeds.to("cuda", torch.float16)
        pipe.negative_prompt_embeds = pipe.negative_prompt_embeds.to("cuda", torch.float16)
        pipe.prompt_embeds          = pipe.prompt_embeds.to("cuda", torch.float16)
        
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        #with pipe.progress_bar(total=num_inference_steps*progress_scale) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue
    
                pipe._current_timestep = t
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])
                
                #Do not worry, the largest of these is 6 MB
                latent_model_input     = latent_model_input.to("cuda", torch.float16)
                timestep               = timestep.to("cuda", torch.float16)
                    
                if pipe.do_classifier_free_guidance:
                    # -- 1. build a virtual 2× batch --------------------------------
                    latent_in  = latent_model_input.expand(2, *latent_model_input.shape[1:])
                    img_embeds = image_embeds.expand(2, *image_embeds.shape[1:])
                    txt_embeds = torch.cat([pipe.negative_prompt_embeds, pipe.prompt_embeds], dim=0)
                    timestep_in = timestep.expand(2)          # shape (2,)
                
                    # -- 2. single transformer call ---------------------------------
                    noise_pair = pipe.transformer(
                        hidden_states              = latent_in,
                        timestep                   = timestep_in,
                        encoder_hidden_states      = txt_embeds,
                        encoder_hidden_states_image= img_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]                                       # (2, C, ...)
                    #vram_report(f"step{i:02d}-post", roots=(pipe.transformer,))
                    
                    noise_uncond, noise_pred = noise_pair.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                    #UNTESTED VRAM SAVINGS, if pipeline breaks, this may be to blame:
                    del noise_pair, noise_uncond, latent_in, img_embeds, txt_embeds, timestep_in
                else:
                    print("WARNING! pipe.do_classifier_free_guidance == false")
                    raise
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=pipe.prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
    
                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                                        
                #Custom VRAM saving code
                del noise_pred
                torch.cuda.empty_cache()
    
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)
    
                    latents = callback_outputs.pop("latents", latents)
                    pipe.prompt_embeds = callback_outputs.pop("prompt_embeds", pipe.prompt_embeds)
                    pipe.negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", pipe.negative_prompt_embeds)
    
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
    
                #if XLA_AVAILABLE:
                #    xm.mark_step()
        
        pipe._current_timestep = None
        
        #VRAM de-frag, move to cpu then back to gpu
        #del prompt_embeds, negative_prompt_embeds          # text
        del image_embeds                                   # CLIP-vision
        del condition                                      # extra latent half
        del latent_model_input, timestep                   # last-step scratch
        latents = latents.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()        # cleans Python refs so allocator can truly release
    
        #free the prefetch gpu_block so that decode has plenty of VRAM available
        pipe.transformer.custom_wan_offload()
    
        selected_latents = {}
        if not output_type == "latent":
            pipe.set_progress_bar_config(desc="Decode latent variables", mininterval=0.0, miniters=1)
            with pipe.progress_bar(total=3) as progress_bar:
                pipe.vae.to(pipe.cuda_device, dtype=torch.float16)
                
                latents = latents.to(pipe.vae.device, pipe.vae.dtype)
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                                        
                #Custom VRAM saving code
                #print(latents_std)
                #print(latents_mean)
                #del latents_std
                #del latents_mean
                #torch.cuda.empty_cache()
                
                progress_bar.update()

                if not IS_WINDOWS:
                    latents = latents.contiguous()
                    gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # ------------- NEW: collect selected raw latents -------------
                #print(latents.shape, latents.dtype, latents.device)
                if save_lossless_frames is not None:
                    # latents shape: (B, C, T', H', W')
                    for idx in save_lossless_frames:
                        t_lat = idx // pipe.vae_scale_factor_temporal   # latent-time index
                        selected_latents[idx] = latents[:, :, t_lat].detach().cpu()
                # --------------------------------------------------------------
                video = pipe.vae.decode(latents, return_dict=False)[0]       #this part takes the longest
                
                progress_bar.update()
                
                video = pipe.video_processor.postprocess_video(video, output_type=output_type)     #this also takes significant time
                
                pipe.vae.to("cpu", dtype=torch.float32)
                pipe.vae.clear_cache()
                del latents
                torch.cuda.empty_cache()
                
                progress_bar.update()
        else:
            video = latents
    
        # Offload all models
        pipe.maybe_free_model_hooks()
    
        if not return_dict:
            return (video, selected_latents)
    
        return (WanPipelineOutput(frames=video), selected_latents)

def wan_vace_custom_pipeline(
    pipe,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    video: Optional[List[PipelineImageInput]] = None,
    mask: Optional[List[PipelineImageInput]] = None,
    reference_images: Optional[List[PipelineImageInput]] = None,
    conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    seed = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    save_lossless_frames: Optional[List[int]] = None,   # <-- NEW
    latent_overrides: Optional[List[int]] = [],   # <-- NEW
):
    with torch.inference_mode():
        if generator == None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Simplification of implementation for now
        if not isinstance(prompt, str):
            raise ValueError("Passing a list of prompts is not yet supported. This may be supported in the future.")
        if num_videos_per_prompt != 1:
            raise ValueError(
                "Generating multiple videos per prompt is not yet supported. This may be supported in the future."
            )

        # 1. Check inputs. Raise error if not correct
        pipe.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            video,
            mask,
            reference_images,
        )

        if num_frames % pipe.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {pipe.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        pipe._guidance_scale   = guidance_scale
        pipe._attention_kwargs = attention_kwargs
        pipe._current_timestep = None
        pipe._interrupt        = False

        device = pipe._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        vae_dtype = pipe.vae.dtype
        transformer_dtype = pipe.transformer.dtype

        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [conditioning_scale] * len(pipe.transformer.config.vace_layers)
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(pipe.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match number of layers {len(pipe.transformer.config.vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if isinstance(conditioning_scale, torch.Tensor):
            if conditioning_scale.size(0) != len(pipe.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {conditioning_scale.size(0)} does not match number of layers {len(pipe.transformer.config.vace_layers)}."
                )
            conditioning_scale = conditioning_scale.to(device=device, dtype=transformer_dtype)

        # 3. Encode input prompt
        if pipe.prompt != prompt or pipe.negativePrompt != negative_prompt:
            pipe.prompt = prompt
            pipe.negativePrompt = negative_prompt
            pipe.set_progress_bar_config(desc="Pre-encoding text prompt")
            with pipe.progress_bar(total=1) as progress_bar:
                pipe.prompt_embeds, pipe.negative_prompt_embeds = pipe.encode_prompt(
                    prompt                      = prompt,
                    negative_prompt             = negative_prompt,
                    #do_classifier_free_guidance = pipe.do_classifier_free_guidance,
                    do_classifier_free_guidance = True,                      #throws an exception if you try to access the member variable for some reason
                    num_videos_per_prompt       = num_videos_per_prompt,
                    prompt_embeds               = None,
                    negative_prompt_embeds      = None,
                    max_sequence_length         = max_sequence_length,
                    device                      = "cpu",
                )
                progress_bar.update()
        else:
            pipe.prompt_embeds, pipe.negative_prompt_embeds = pipe.encode_prompt(
                prompt                      = prompt,
                negative_prompt             = negative_prompt,
                do_classifier_free_guidance = pipe.do_classifier_free_guidance,
                num_videos_per_prompt       = num_videos_per_prompt,
                prompt_embeds               = pipe.prompt_embeds,
                negative_prompt_embeds      = pipe.negative_prompt_embeds,
                max_sequence_length         = max_sequence_length,
                device                      = device,
            )

        pipe.prompt_embeds = pipe.prompt_embeds.to(transformer_dtype)
        if pipe.negative_prompt_embeds is not None:
            pipe.negative_prompt_embeds = pipe.negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # 5. Prepare latent variables
        pipe.set_progress_bar_config(desc="Prepare latent variables")
        pipe.vae.to(pipe.cuda_device, dtype=torch.float32)
        with pipe.progress_bar(total=2) as progress_bar:
            video, mask, reference_images = pipe.preprocess_conditions(
                video,
                mask,
                reference_images,
                batch_size,
                height,
                width,
                num_frames,
                torch.float32,
                device,
            )
            num_reference_images = len(reference_images[0])
            
            print("--- video.shape --------")
            print(video.shape)

            #if things aren't on cuda then we get an exception:
            #  NotImplementedError: Could not run 'aten::slow_conv3d_forward' with arguments from the 'CUDA' backend.
            video            = video.to("cuda", dtype=torch.float32)
            mask             = mask.to("cuda", dtype=torch.float32)
            for batch in reference_images:          # outer list = batch dimension
                for i, img in enumerate(batch):     # inner list = ref images per video
                    batch[i] = img.to("cuda", dtype=torch.float32)
            conditioning_latents = pipe.prepare_video_latents(
                video, mask, reference_images, generator, device="cuda"
            )
            print("--- conditioning_latents.shape")
            print(conditioning_latents.shape)
            if len(latent_overrides) > 0:
                z_dim   = pipe.vae.config.z_dim                  # 8
                t_scale = pipe.vae_scale_factor_temporal         # 2 (default)
                ref_ch  = len(reference_images[0]) * 2 * z_dim   # 16 if one ref, else 0
                
                print("--- latent_overrides[0]")
                print(latent_overrides[0][1].shape)

                for f_idx, latent_slice in latent_overrides:
                    t_idx = f_idx // t_scale                     # which latent frame
                    ch0   = ref_ch                               # inactive slice start
                    conditioning_latents[0, ch0 : ch0 + z_dim, t_idx] = latent_slice.to(
                        device=conditioning_latents.device,
                        dtype =conditioning_latents.dtype
                    )
                    # optional: zero the companion reactive slice
                    conditioning_latents[0, ch0 + z_dim : ch0 + 2*z_dim, t_idx] = 0

            mask = pipe.prepare_masks(mask, reference_images, generator)
            conditioning_latents = torch.cat([conditioning_latents, mask], dim=1)
            conditioning_latents = conditioning_latents.to(transformer_dtype)
            progress_bar.update()

            num_channels_latents = pipe.transformer.config.in_channels
            latents = pipe.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames + num_reference_images * pipe.vae_scale_factor_temporal,
                torch.float32,
                device,
                generator,
                latents,
            )
            print("--- latents.shape ------")
            print(latents.shape)
            pipe.vae.to("cpu", dtype=torch.float32)
            #We send these right back to cuda very soon, maybe dont put them on cpu here?
            latents = latents.to("cpu", dtype=torch.float32)
            conditioning_latents = conditioning_latents.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
            progress_bar.update()

        if conditioning_latents.shape[2] != latents.shape[2]:
            logger.warning(
                "The number of frames in the conditioning latents does not match the number of frames to be generated. Generation quality may be affected."
            )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        pipe._num_timesteps = len(timesteps)
                
        pipe.set_progress_bar_config(desc="", mininterval=0.0, miniters=1)
        
        if pipe.transformer.enable_magcache and pipe.transformer.mag_ratio_curve is None:
            raise
            fingerprint = fingerprint_from_string(pipe.model_name + pipe.embedding_string + pipe.lora_string + "|" + str(num_inference_steps) + "|" + str(num_frames) + "|" + str(height) + "x" + str(width))
            if pipe.num_inference_steps is not None:
                assert pipe.num_inference_steps == num_inference_steps, "magcache isn't calibrated for variable sample_steps"
                pipe.num_inference_steps = num_inference_steps
            latents, conditioning_latents = build_magcache_if_needed(
                pipe,
                latents, conditioning_latents,
                timesteps,
                pipe.prompt_embeds, pipe.negative_prompt_embeds, image_embeds,
                sample_steps=num_inference_steps,
                cache_dir="magcache_json",
                fingerprint=fingerprint,
                transformer_dtype=transformer_dtype,
                attention_kwargs=attention_kwargs,
                guidance_scale=guidance_scale
            )
        
        #make sure we have our prefetch gpu_block ready
        #pipe.transformer.custom_wan_onload(use_sageattention=pipe.use_sageattention)
    
        if pipe.transformer.enable_magcache:
            pipe.transformer.reset_magcache_state()
            
        #vram_report(f"before setup")
        
        #Do not worry, the largest of these is 6 MB
        #del image_embeds                                   # CLIP-vision
        pipe.negative_prompt_embeds = pipe.negative_prompt_embeds.to("cuda", torch.float16)
        pipe.prompt_embeds          = pipe.prompt_embeds.to("cuda", torch.float16)
                
        conditioning_latents        = conditioning_latents.to("cuda", torch.float16)
        #encoder_hidden_states       = encoder_hidden_states.to("cuda", torch.float16)
        #encoder_hidden_states_image = encoder_hidden_states_image.to("cuda", torch.float16)
        
        
        #vram_report(f"before steps")

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                pipe._current_timestep = t
                #latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])
                
                #Do not worry, the largest of these is 6 MB
                latent_model_input     = latents.to("cuda", torch.float16)
                timestep               = timestep.to("cuda", torch.float16)
                
                if pipe.do_classifier_free_guidance:
                    # -- 1. build a virtual 2× batch --------------------------------
                    latent_in       = latent_model_input.expand(2, *latent_model_input.shape[1:])
                    conditioning_in = conditioning_latents.expand(2, *conditioning_latents.shape[1:])
                    #img_embeds      = image_embeds.expand(2, *image_embeds.shape[1:])
                    txt_embeds      = torch.cat([pipe.negative_prompt_embeds, pipe.prompt_embeds], dim=0)
                    timestep_in     = timestep.expand(2)          # shape (2,)
                    
                    noise_pair = pipe.transformer(
                        hidden_states         = latent_in,
                        timestep              = timestep_in,
                        encoder_hidden_states = txt_embeds,
                        control_hidden_states = conditioning_in,
                        control_hidden_states_scale = conditioning_scale,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    
                    noise_uncond, noise_pred = noise_pair.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                    #UNTESTED VRAM SAVINGS, if pipeline breaks, this may be to blame:
                    del noise_pair, noise_uncond, latent_in, conditioning_in, txt_embeds, timestep_in
                else:
                    print("WARNING! pipe.do_classifier_free_guidance == false")
                    raise
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=pipe.prompt_embeds,
                        control_hidden_states=conditioning_latents,
                        control_hidden_states_scale=conditioning_scale,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                                        
                #Custom VRAM saving code
                del noise_pred
                torch.cuda.empty_cache()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    pipe.prompt_embeds = callback_outputs.pop("prompt_embeds", pipe.prompt_embeds)
                    pipe.negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", pipe.negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

                #if XLA_AVAILABLE:
                #    xm.mark_step()
        
        pipe._current_timestep = None
        
        #VRAM de-frag, move to cpu then back to gpu
        del conditioning_latents, conditioning_scale
        del latent_model_input, timestep                   # last-step scratch
        latents = latents.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()        # cleans Python refs so allocator can truly release
    
        #free the prefetch gpu_block so that decode has plenty of VRAM available
        #pipe.transformer.custom_wan_offload()
    
        selected_latents = {}
        if not output_type == "latent":
            pipe.set_progress_bar_config(desc="Decode latent variables", mininterval=0.0, miniters=1)
            with pipe.progress_bar(total=3) as progress_bar:
                pipe.vae.to(pipe.cuda_device, dtype=torch.float16)
                
                latents = latents.to(pipe.vae.device, pipe.vae.dtype)
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean
                                        
                #Custom VRAM saving code
                #print(latents_std)
                #print(latents_mean)
                #del latents_std
                #del latents_mean
                #torch.cuda.empty_cache()
                
                progress_bar.update()

                if not IS_WINDOWS:
                    latents = latents.contiguous()
                    gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # ------------- NEW: collect selected raw latents -------------
                #print(latents.shape, latents.dtype, latents.device)
                if save_lossless_frames is not None:
                    # latents shape: (B, C, T', H', W')
                    for idx in save_lossless_frames:
                        t_lat = idx // pipe.vae_scale_factor_temporal   # latent-time index
                        selected_latents[idx] = latents[:, :, t_lat].detach().cpu()
                # --------------------------------------------------------------
                video = pipe.vae.decode(latents, return_dict=False)[0]       #this part takes the longest
                
                progress_bar.update()
                
                video = pipe.video_processor.postprocess_video(video, output_type=output_type)     #this also takes significant time
                
                pipe.vae.to("cpu", dtype=torch.float32)
                pipe.vae.clear_cache()
                del latents
                torch.cuda.empty_cache()
                
                progress_bar.update()
        else:
            video = latents
    
        # Offload all models
        pipe.maybe_free_model_hooks()
    
        if not return_dict:
            return (video, selected_latents)
    
        return (WanPipelineOutput(frames=video), selected_latents)


def get_duplicate_keyframe_indices(gen_len, key_indices):
    """
    Returns a list of frame indices (from generated clip) that should be
    trimmed because they're perfect duplicates of the neighbouring video.
    Currently: drop if the keyframe lies at idx 0 or idx == gen_len-1.
    """
    to_trim = []
    left_edge = 0
    right_edge = (gen_len - 1)
    while left_edge in key_indices:
        to_trim.append(left_edge)
        left_edge += 1
    while right_edge in key_indices:
        to_trim.append(right_edge)
        right_edge -= 1
    return to_trim
    
def generate_and_publish(
    pipe,
    pipe_kwargs,
    start_image_path = None,
    start_image_size = None,
    prior_video_path = "",
    keyframe_idxs = [],
    out_path = "",
    job_path = "",
    fps = 16,
    quality = 5,
    apply_color_match = True,
    do_keyframe_trim = True,
    publish = True,
):
    if isinstance(pipe, WanImageToVideoPipeline):
        assert start_image_path or generate_and_publish.start_image, "Must supply an initial image for i2v generation"
        #Prepare image if we haven't already
        if not hasattr(generate_and_publish, 'start_image_path') or start_image_path != generate_and_publish.start_image_path:
            generate_and_publish.start_image_path = start_image_path
            generate_and_publish.start_image = resize_image_for_wan_480p(load_image_from_file(start_image_path), start_image_size)
        pipe_kwargs['image']  = generate_and_publish.start_image
        pipe_kwargs['height'] = generate_and_publish.start_image.height
        pipe_kwargs['width']  = generate_and_publish.start_image.width
    
    if not hasattr(generate_and_publish, 'job_script') and job_path != "":
        with open(job_path, 'r') as file:
            generate_and_publish.job_script = file.read()
            
    if prior_video_path != "":
        result = load_image_from_file(prior_video_path, frameN=-1)
        assert isinstance(result, list)
        image64_frames = []
        for frame in result:
            image64_frames.append(image_to_base64(frame, lossless_mode=True, img_format="PNG"))
        all_videos_data: List[Dict] = [{
            "video_base64"  : load_image_from_file(prior_video_path, frameN=-1, convert="base64"),
            "image_frames"  : image64_frames,
            "frame_latents" : {},
            "metadata"      : {
                "num_frames": len(image64_frames),
            }
        }]
        #remove duplicate frames at the edges of the video
        trim_frames = get_duplicate_keyframe_indices(pipe_kwargs["num_frames"], keyframe_idxs)
    else:
        trim_frames = []
    
    generator_states = []
    generator = torch.Generator(device="cpu").manual_seed(pipe_kwargs['seed'])
    pipe_kwargs['generator'] = generator
    
    if isinstance(pipe, WanImageToVideoPipeline):
        frames, frame_latents = wan_i2v_custom_pipeline(pipe, **pipe_kwargs)
    elif isinstance(pipe, WanVACEPipeline):
        frames, frame_latents = wan_vace_custom_pipeline(pipe, **pipe_kwargs)
    output = frames.frames[0]
    
    #print(type(output))        # <class 'numpy.ndarray'>
    #print(type(output[0]))     # <class 'numpy.ndarray'>
    # ------------ BEGIN: weighted colour match -----------------------------------
    if apply_color_match and keyframe_idxs:
        # --- 1. palette: mu/std per key-frame (Lab) ---------------------------------
        key_stats = {}
        for k in keyframe_idxs:
            pil_img = pipe_kwargs["video"][k]                     # PIL RGB
            lab_k   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB).astype(np.float32)
            flat    = lab_k.reshape(-1, 3)
            key_stats[k] = (flat.mean(0), flat.std(0))            # (mu_k, std_k)

        # --- 2. indices to grade (non-key frames) --------------------------------
        to_grade = sorted(set(range(len(output))) - set(keyframe_idxs))

        # --- 3. create a uint8 BGR copy for those frames -------------------------
        frames_bgr_u8 = []
        for idx in to_grade:
            frame = output[idx]
            if frame.dtype != np.uint8:
                frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame_u8 = frame.copy()
            frames_bgr_u8.append(frame_u8[..., ::-1])            # RGB->BGR

        weight_pow = 1.0   # 1 = linear fall-off, 2 = quadratic, etc.
        chroma_min, chroma_max = 0.7, 1.3

        # --- 4. process each frame with distance-weighted mu/std --------------------
        for list_pos, idx in enumerate(to_grade):
            bgr = frames_bgr_u8[list_pos]
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

            mu_s, std_s = lab.reshape(-1,3).mean(0), lab.reshape(-1,3).std(0)

            # blended target mu/std
            mu_t  = np.zeros(3, np.float32)
            std_t = np.zeros(3, np.float32)
            w_sum = 0.0
            for k, (mu_k, std_k) in key_stats.items():
                w = 1.0 / ((abs(idx - k) + 1e-4) ** weight_pow)
                mu_t  += w * mu_k
                std_t += w * std_k
                w_sum += w
            mu_t  /= w_sum
            std_t /= w_sum

            # scale / shift with chroma clamp
            scale = std_t / (std_s + 1e-5)
            scale[1:] = np.clip(scale[1:], chroma_min, chroma_max)
            lab[:] = (lab - mu_s) * scale + mu_t

            # write back to original tensor (RGB, same dtype)
            out_rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                                   cv2.COLOR_LAB2RGB)
            if output.dtype != np.uint8:
                out_rgb = out_rgb.astype(np.float32) / 255.0
            output[idx] = out_rgb
    # ------------ END: weighted colour match -------------------------------------

    if do_keyframe_trim:
        for idx in reversed(trim_frames):
            output = np.delete(output, idx, axis=0)
            
    #save the initial .mp4
    unix_time = int(time.time())
    out_path = FILE_PREPEND / out_path.format(unix_time=unix_time)
    print(out_path)
    ensure_folder_exists(out_path)
    export_to_video(output, out_path, fps=fps, quality=quality)  #10 is highest quality, 5 is default

    pil_images = extract_frames(out_path)
    image64_frames = []
    for frame in pil_images:
        image64_frames.append(image_to_base64(frame, lossless_mode=True, img_format="PNG"))

    base64_video = mp4_to_base64(out_path)

    generator_states.append(generator_to_b64(generator))

    metadata = {
        "Prompt":            pipe_kwargs['prompt'],
        "Negative Prompt":   pipe_kwargs['negative_prompt'],
        "Seed":              pipe_kwargs['seed'],
        "vid_width":         pipe_kwargs['width'],
        "vid_height":        pipe_kwargs['height'],
        "Steps":             pipe_kwargs['num_inference_steps'],
        "num_frames":        pipe_kwargs['num_frames'],
        #"Strength":          inpainting_strength,
        "Guidance Scale":    pipe_kwargs['guidance_scale'],
        "LoRAs":             pipe.lora_string + " " + pipe.embedding_string,
        #"Mask":              inpainting_mask_file,
        "Mask":              "",
        "Model":             pipe.model_name,
        "Mode":              "Image2Video",
        "InputImage1":       start_image_path,
        "InputImage2":       "",
        "GeneratorStates":   generator_states,
        "FPS":               fps,
    }
    videos_data = [{
        "video_base64"  : base64_video,
        "image_frames"  : image64_frames,
        "frame_latents" : frame_latents,
        "metadata"      : metadata
    }]
    
    if publish:
        if prior_video_path != "":
            all_videos_data.append(videos_data[0])

            html_content = generate_video_html(all_videos_data, PYTHON_SCRIPT, generate_and_publish.job_script)
            out_path = os.path.splitext(out_path)[0] + "_concatenated.html"
            save_html_file(html_content, out_path)

            print(f"[CHAIN] concatenated HTML saved as {out_path}")
        else:
            html_content = generate_video_html(videos_data, PYTHON_SCRIPT, generate_and_publish.job_script)
            out_path = os.path.splitext(out_path)[0] + ".html"
            save_html_file(html_content, out_path)
            
            print(f"[VIDEO]: HTML file saved as {out_path}")
            
        videos_data[0]["path"] = out_path
    
    return videos_data
    

###############################################################################
# LOAD EMBEDDINGS and LoRAs
###############################################################################
def import_loras(pipe,
                 lora_paths = [],
                 lora_weights = [],
                 embedding_paths = [],
                 embedding_tokens = [],
                 embedding_replace = []):
    
    if not IS_WINDOWS:
        embedding_paths = [s.replace('\\', '/') for s in embedding_paths]
        lora_paths = [s.replace('\\', '/') for s in lora_paths]
        
    #pipe = pipe.to("cpu")                #pipe is assumed to already be on cpu
    with torch.inference_mode():
        #embedding_paths = []
        #embedding_tokens = []
        #embedding_replace = [] # tokens to be replaced, pick rare words you wont use
        embedding_i = 0
        pipe.embedding_string = "{"
        for embedding_path in tqdm(embedding_paths, desc="Loading Embeddings"):
            pipe = hacky_insert_multivector_embedding(pipe, embedding_path, embedding_tokens[embedding_i], embedding_replace[embedding_i])
            pipe.embedding_string += embedding_path + " : " + str(embedding_replace[embedding_i]) + ", "
            embedding_i += 1
        pipe.embedding_string += "}"
        
        lora_paths   = ensure_2d(lora_paths)
        lora_weights = ensure_2d(lora_weights)
    
        pipe.lora_string = "{"
        for inner_lora_paths, inner_lora_weights in zip(lora_paths, lora_weights):
            #lora_paths = []
            #lora_weights = []
            lora_names = []
            lora_i = 0
            pipe.lora_string += "{"
            for lora in tqdm(inner_lora_paths, desc="Loading LoRAs"):
                pipe.load_lora_weights(lora, adapter_name=f"lora_{lora_i}")
                #pipe.load_lora_weights(lora, prefix=None, adapter_name=f"lora_{lora_i}")        #I made this change to try loading a troublesome LoRA, I dont know if this will have bad consequences
                lora_names += [f"lora_{lora_i}"]
                pipe.lora_string += lora + " : " + str(inner_lora_weights[lora_i]) + ", "
                lora_i += 1
            pipe.lora_string += "}"
        
            #Fuse LoRAs
            if len(inner_lora_paths) > 0:
                for _ in tqdm(range(1), desc="Fusing LoRAs"):
                    pipe.set_adapters(lora_names, adapter_weights=inner_lora_weights)
                    pipe.fuse_lora(adapter_names=lora_names, lora_scale=1.0)
                    pipe.unload_lora_weights()
        pipe.lora_string += "}"
        
    torch.cuda.synchronize()
    return pipe


_bootstrap()      # still execute when run as script
if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    #_bootstrap()                     # ensure env is ready
    #main()
