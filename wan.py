try:
    import os, argparse, gc, math, sys
    from pathlib import Path
    import platform
    import time
    import imageio
    import re
    import base64
    import cv2
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
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    from diffusers.utils import load_image, export_to_video
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union
    import runpy
    import json
    import logging
    from html import escape
    import traceback
    import getpass
except ModuleNotFoundError as e:
    sys.stderr.write('Required module not found. Did you forget to activate the virtual environment?\n')
    sys.stderr.write(f'Original exception: {e}\n')
    sys.exit(1)
    
IS_WINDOWS = platform.system() == "Windows"

# Set temp dirs appropriately
SCRIPT_DIR = Path(__file__).resolve().parent
if IS_WINDOWS:
    print("Running on Windows")
    os.environ["TEMP"] = r"C:\Temp"
    os.environ["TMP"] = r"C:\Temp"
    MODEL_DIR_I2V = Path(r"C:\Wan2_1\wan_i2v")
    FILE_PREPEND = ""
    
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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"        #Is this real? ChatGPT told me to set it due to OOM issues involving VRAM fragmentation
    torch._dynamo.config.recompile_limit = 256
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["TEMP"] = "/tmp"
    os.environ["TMP"] = "/tmp"

    username = getpass.getuser()
    MOUNT_POINT = Path(f"/media/{username}/Weights")
    if not os.path.ismount(MOUNT_POINT):
        sys.stderr.write(f'Error: Drive is not mounted at {MOUNT_POINT}\n')
        sys.exit(1)
    MODEL_DIR_I2V = MOUNT_POINT / "Wan2_1" / "wan_i2v"
    FILE_PREPEND = SCRIPT_DIR

REPO_ID_I2V = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

fileType="WEBP"
htmlImageType="image/webp"

oneTimeDebugStatement = True

torch.set_num_threads(8)            # physical cores
torch.set_num_interop_threads(1)
                      
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


def ensure_2d(arr):
    return [arr] if isinstance(arr, list) and not any(isinstance(sub, list) for sub in arr) else arr

def mod_floor(x, m):       # nearest multiple of m not exceeding x
    return x // m * m
    
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
    remove_duplicates=True,
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
            base64_data = video_match.group(1)
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

def ensure_folder_exists(filename):
    """Save the HTML content to a file."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# Function to create a thumbnail from base64 data
def create_thumbnail(base64_data, max_width, max_height):
    # Decode the base64 data
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data))

    # Resize the image while maintaining the aspect ratio
    image.thumbnail((max_width, max_height))
    width, height = image.size

    # Convert the image back to base64
    buffered = BytesIO()
    image.save(buffered, format="WEBP")
    thumbnail_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:{htmlImageType};base64,{thumbnail_base64}", width, height

def frames_to_mp4_base64(frames, fps=8):
    """Save MP4 to a temp file, read it into memory, then delete the file."""
    temp_filename = "temp_video.mp4"
    
    # Save as file first (imageio will use its internal FFmpeg)
    imageio.mimsave(temp_filename, frames, fps=fps)
    
    # Read the file into memory
    with open(temp_filename, "rb") as f:
        mp4_data = f.read()

    # Cleanup the temp file
    os.remove(temp_filename)

    # Return Base64-encoded MP4
    return base64.b64encode(mp4_data).decode("utf-8")
    
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
    
lossless_mode = False
def image_to_base64(image):
    """Convert a PIL Image to a Base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=fileType, lossless=lossless_mode)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def generate_video_html(videos_data, python_script, job_script):
    """Generate an HTML string with embedded images, metadata, and the original Python script."""
    html_content = "<html><head><title>Generated Videos</title></head><body>"
    
    vid_id = 0
    img_id = 0
    for data in reversed(videos_data):
        vid_id += 1
        base64_video     = data['video_base64']
        vid_width        = data['metadata']['vid_width']
        vid_height       = data['metadata']['vid_height']
        img_frames       = data['image_frames']
        prompt           = data['metadata']['Prompt']
        negative_prompt  = data['metadata']['Negative Prompt']
        seed             = data['metadata']['Seed']
        steps            = data['metadata']['Steps']
        guidance_scale   = data['metadata']['Guidance Scale']
        loras            = data['metadata']['LoRAs']
        model_path       = data['metadata']['Model']
        metadata_mode    = data['metadata']["Mode"]
        generatorStates  = data['metadata']['GeneratorStates']
        fps              = data['metadata']["FPS"]
        input_image_path = data['metadata']['InputImage1']

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
            <div style="display: none; justify-content: left; flex-wrap: wrap;" id="frames_div">"""
        
        for image64 in img_frames:
            input_image_thumbnail_data, input_w, input_h = create_thumbnail(image64, vid_width / 4, vid_height / 4)
            html_content += f"""
                <img id="generated_{img_id}" src="{input_image_thumbnail_data}" alt="Frame {img_id}" title="Frame {img_id}" onclick="jumpToFrame(this)" />
            """
            img_id += 1
            
        html_content += f"""
            </div>
            """
        
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
        <pre>{python_script}</pre>
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
# DEBUG FUNCTIONS
###############################################################################
def tensor_size_mb(t: torch.Tensor, name) -> float:
    t_size =  t.numel() * t.element_size() / 1024**2
    print(f"{name}   {t_size:6.1f} MB")
    
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
    for blk in pipe.transformer.blocks:                   #This should be done after enable_xformers_memory_efficient_attention
        blk.attn1.processor = WanXFormersSelfAttnProcessor(use_sageattention=use_sageattention)      #Confirmed via print statements
        blk.attn2.processor = WanXFormersCrossAttnProcessor(use_sageattention=use_sageattention)
        #print(blk.attn1.add_k_proj)
        #print(blk.attn2.add_k_proj)
        assert blk.attn1.add_k_proj is None,     "blk.attn1.add_k_proj != None, code change required"
        assert blk.attn2.add_k_proj is not None, "blk.attn2.add_k_proj == None, code change required"
    
    #torch.backends.cuda.enable_mem_efficient_sdp(True)
    #pipe.enable_sequential_cpu_offload()   # GPU‑related
    #pipe.enable_model_cpu_offload()        # GPU‑related
    pipe.enable_attention_slicing()
    
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
        
        # Insert TeaCache cache decision logic here.
        # E.g. calculate modulated input, compare with previous, decide to skip or not.
        # decide reuse?
        if self.enable_teacache:
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
            self.cnt = (self.cnt + 1) % 1000
        else:
            do_compute = True
            
            
        if not do_compute:
            if is_even:
                assert self.prev_res_even is not None, "prev_res_even is None but computation was skipped"
                hidden_states += self.prev_res_even
            else:
                assert self.prev_res_odd is not None, "prev_res_odd is None but computation was skipped"
                hidden_states += self.prev_res_odd
        else:
            if self.enable_teacache:
                ori_x = hidden_states.clone()
                
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                print("WARNING! This codepath is not up to date and may fail in unknown ways!")
                raise NotImplementedError("Gradient checkpointing not currently supported in this script.")
                for block in self.blocks:
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

                    #Fast in-place tensor copy  CPU→GPU   (src pinned on CPU, dest already on GPU).
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
                        hs  = hidden_states[ b : b+1 ]                    # (1, …)
                        enc = encoder_hidden_states[ b : b+1 ]
                        tpr = timestep_proj[ b : b+1 ]

                        out = active(hs, enc, tpr, rotary_emb)

                        if IS_WINDOWS:
                            out_parts.append(out)
                        else:
                            out_parts.append(out.clone())
                        #out_parts.append(out)

                    hidden_states = torch.cat(out_parts, dim=0)           # (2, …)
                    del out_parts
                    torch.cuda.empty_cache()
                    #maybe fix bluescreen?
                    torch.cuda.synchronize()

                    # ---- record "my kernels queued" event ----
                    self.done_lock = torch.cuda.Event(enable_timing=False)
                    self.done_lock.record()                               # on stream-0
                    
            if self.enable_teacache:
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

# This is called once ever
def custom_wan_init(self, teacache_threshold = 0.0):
    self.gpu_blocks = []
    
    if teacache_threshold > 0.0:
        self.enable_teacache    = True
        self.teacache_threshold = teacache_threshold
        self.use_ref_steps      = True
        self.reset_teacache_state()
    else:
        self.enable_teacache    = False
    
    self.custom_init_done = True
    print("Creating WanTransformer locks now, you should only see this happen once")
    self.prefetch_stream = torch.cuda.Stream()      #Only created once
    self.done_lock = torch.cuda.Event(enable_timing=False) #This is a one time dummy lock that we create, but we will recreate it many times
    self.done_lock.record()                                # on stream-0

#Needs to be called before the transformer can be used, call right before inference
def custom_wan_onload(self, *, use_sageattention=False):
    def make_gpu_worker(template):
        with torch.inference_mode():
            blk = copy.deepcopy(template).to("cuda", dtype=torch.float16)
            #Unsure if its necessary to recreate the processors, but its probably needed since we're compiling them
            blk.attn1.processor = WanXFormersSelfAttnProcessor(use_sageattention=use_sageattention)
            blk.attn2.processor = WanXFormersCrossAttnProcessor(use_sageattention=use_sageattention)
            if IS_WINDOWS:
                return blk
            else:
                return torch.compile(blk,
                                 mode="reduce-overhead",
                                 dynamic=True, fullgraph=False)
    self.gpu_blocks = [
        make_gpu_worker(self.blocks[0]),
        make_gpu_worker(self.blocks[1])
    ]
    torch.cuda.synchronize()

#Use this after you are done with the transformer to free up VRAM, call right after inference
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
    
        
WanTransformer3DModel.forward              = custom_wan_forward
WanTransformer3DModel.reset_teacache_state = reset_teacache_state
WanTransformer3DModel.custom_wan_init      = custom_wan_init
WanTransformer3DModel.custom_wan_onload    = custom_wan_onload
WanTransformer3DModel.custom_wan_offload   = custom_wan_offload
    
from diffusers.image_processor import PipelineImageInput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
def wani2v_custom_pipeline(
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
    use_sageattention: bool = False,
):
    with torch.inference_mode():
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
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt                      = prompt,
            negative_prompt             = negative_prompt,
            do_classifier_free_guidance = pipe.do_classifier_free_guidance,
            num_videos_per_prompt       = num_videos_per_prompt,
            prompt_embeds               = prompt_embeds,
            negative_prompt_embeds      = negative_prompt_embeds,
            max_sequence_length         = max_sequence_length,
            device                      = device,
        )
    
        # Encode image embedding
        transformer_dtype = pipe.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        
        if image_embeds is None:
            image_embeds = pipe.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)
    
        # 4. Prepare timesteps
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps
    
        # 5. Prepare latent variables
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
                
        pipe.set_progress_bar_config(desc="", mininterval=0.0, miniters=1)
        
        #make sure we have our prefetch gpu_block ready
        pipe.transformer.custom_wan_onload(use_sageattention=use_sageattention)
        
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue
    
                pipe._current_timestep = t
                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])
                
                latent_model_input     = latent_model_input.to("cuda", torch.float16)
                image_embeds           = image_embeds.to("cuda", torch.float16)
                negative_prompt_embeds = negative_prompt_embeds.to("cuda", torch.float16)
                prompt_embeds          = prompt_embeds.to("cuda", torch.float16)
                timestep               = timestep.to("cuda", torch.float16)
                #Do not worry, the largest of these is 6 MB
                #global oneTimeDebugStatement
                #if oneTimeDebugStatement:
                #    oneTimeDebugStatement = False
                #    tensor_size_mb(latent_model_input, "latent_model_input size")
                #    tensor_size_mb(image_embeds, "image_embeds size")
                #    tensor_size_mb(timestep, "timestep size")
                    
                if pipe.do_classifier_free_guidance:
                    # -- 1. build a virtual 2× batch --------------------------------
                    latent_in  = latent_model_input.expand(2, *latent_model_input.shape[1:])
                    img_embeds = image_embeds.expand(2, *image_embeds.shape[1:])
                    txt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    timestep_in = timestep.expand(2)          # shape (2,)
                
                    # -- 2. single transformer call ---------------------------------
                    #with cuda_scope(f"denoise step {i:02d}  call-transformer"):
                    noise_pair = pipe.transformer(
                        hidden_states              = latent_in,
                        timestep                   = timestep_in,
                        encoder_hidden_states      = txt_embeds,
                        encoder_hidden_states_image= img_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]                                       # (2, C, ...)
                    
                    noise_uncond, noise_pred = noise_pair.chunk(2)
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                else:
                    print("WARNING! pipe.do_classifier_free_guidance == false")
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
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
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
    
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
                
                #tensor_size_mb(latents, "latents decode size")
                progress_bar.update()

                if not IS_WINDOWS:
                    latents = latents.contiguous()
                    gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                #print(latents.shape, latents.dtype, latents.device)
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
            return (video,)
    
        return WanPipelineOutput(frames=video)

def main():
###############################################################################
# PARSE COMMAND LINE
###############################################################################
    ap = argparse.ArgumentParser(
        description="Wan 2.1 image‑to‑video for 8 GB GPUs.")
    ap.add_argument("--install_i2v", action="store_true", help="Download, then exit")
    ap.add_argument('--skip', type=int, help='skip image number', default='0')
    ap.add_argument('--jobpath', type=str, help='path to job variables', default="")
    args = ap.parse_args()

    skipNum = args.skip

    if args.install_i2v:
        MODEL_DIR_I2V.mkdir(parents=True, exist_ok=True)
        download_repo(REPO_ID_I2V, MODEL_DIR_I2V)
        #quantize_clip(MODEL_DIR_I2V)
        print("Install complete.")
        return

    # generation mode
    if not MODEL_DIR_I2V.exists():
        sys.exit("Model directory not found. Run with --install_i2v first.")

###############################################################################
# LOAD THE SCRIPT TEXT
###############################################################################
    python_script = ""
    job_script = ""
    
    with open(__file__, 'r') as file:
        python_script = file.read()
    
    if args.jobpath != "":
        with open(args.jobpath, 'r') as file:
            job_script = file.read()
        job_vars = runpy.run_path(args.jobpath)
    else:
        job_vars = {}
        
###############################################################################
# PIPELINE INITIALIZATION
###############################################################################
    
    model_on_cpu = True
    do_warmup = False
    
    start_image_path          = job_vars.get("image_path",              "")
    start_frame               = job_vars.get("start_frame",             0)
    setName                   = job_vars.get("setName",                 "none")
    folderName                = job_vars.get("folderName",              "none")
    inpainting_mask_file      = job_vars.get("inpainting_mask_file",    "")
    lora_paths                = job_vars.get("lora_paths",             [])
    lora_weights              = job_vars.get("lora_weights",           [])
    embedding_paths           = job_vars.get("embedding_paths",        [])
    embedding_tokens          = job_vars.get("embedding_tokens",       [])
    embedding_replace         = job_vars.get("embedding_replace",      [])
    prompt1                   = job_vars.get("prompt1",                 "")
    promptParts2              = job_vars.get("promptParts2",           [""])
    promptParts3              = job_vars.get("promptParts3",           [""])
    negative_prompt           = job_vars.get("negative_prompt",         "")
    sizes                     = job_vars.get("sizes",                 [[512, 512]])
    guidance_scales           = job_vars.get("guidance_scales",        [5])
    steps_list                = job_vars.get("steps_list",             [50])
    fps                       = job_vars.get("fps",                     16)
    seeds                     = job_vars.get("seeds",                   list(range(1, 65)))
    num_frames                = job_vars.get("num_frames",              81)
    teacache_threshold        = job_vars.get("teacache_threshold",      0.0)
    use_sageattention         = job_vars.get("use_sageattention",       False)
    
    #If you have problems with sageattention in windows, uncomment this
    #if IS_WINDOWS and use_sageattention:
    #    print("Warning: SageAttention is not available in Windows, disabling...")
    #    use_sageattention = False

    pipe = build_pipeline_i2v(use_sageattention=use_sageattention)
    
    if not IS_WINDOWS:
        embedding_paths = [s.replace('\\', '/') for s in embedding_paths]
        lora_paths = [s.replace('\\', '/') for s in lora_paths]

###############################################################################
# LOAD EMBEDDINGS and LoRAs
###############################################################################
    #pipe = pipe.to("cpu")                #pipe is assumed to already be on cpu
    with torch.inference_mode():
        #embedding_paths = []
        embedding_tokens = []
        embedding_replace = [] # tokens to be replaced, pick rare words you wont use
        embedding_i = 0
        embedding_string = "{"
        for embedding_path in tqdm(embedding_paths, desc="Loading Embeddings"):
            pipe = hacky_insert_multivector_embedding(pipe, embedding_path, embedding_tokens[embedding_i], embedding_replace[embedding_i])
            embedding_string += embedding_path + " : " + str(embedding_replace[embedding_i]) + ", "
            embedding_i += 1
        embedding_string += "}"
        
        lora_paths   = ensure_2d(lora_paths)
        lora_weights = ensure_2d(lora_weights)
    
        lora_string = "{"
        for inner_lora_paths, inner_lora_weights in zip(lora_paths, lora_weights):
            #lora_paths = []
            lora_weights = []
            lora_names = []
            lora_i = 0
            lora_string += "{"
            for lora in tqdm(inner_lora_paths, desc="Loading LoRAs"):
                pipe.load_lora_weights(lora, adapter_name=f"lora_{lora_i}")
                #pipe.load_lora_weights(lora, prefix=None, adapter_name=f"lora_{lora_i}")        #I made this change to try loading a troublesome LoRA, I dont know if this will have bad consequences
                lora_names += [f"lora_{lora_i}"]
                lora_string += lora + " : " + str(inner_lora_weights[lora_i]) + ", "
                lora_i += 1
            lora_string += "}"
        
            #Fuse LoRAs
            if len(inner_lora_paths) > 0:
                for _ in tqdm(range(1), desc="Fusing LoRAs"):
                    pipe.set_adapters(lora_names, adapter_weights=inner_lora_weights)
                    pipe.fuse_lora(adapter_names=lora_names, lora_scale=1.0)
                    pipe.unload_lora_weights()
        lora_string += "}"
        
    torch.cuda.synchronize()
    
    #pin the transformer for more efficient memory swapping
    for i, block in tqdm(enumerate(pipe.transformer.blocks), total=len(pipe.transformer.blocks), desc="Pinning transformer blocks"):
        check_cuda_tensors(block)  # Sanity: ensure clean start
        
        pinned_block = pin_module_tensors(block)
        pipe.transformer.blocks[i] = pinned_block  # Replace in-place
        del block
        torch.cuda.empty_cache()
        gc.collect()
        
    pipe.transformer.custom_wan_init(
        teacache_threshold = teacache_threshold
    )
        
    #pipe = pipe.to(device_str)
    #torch.cuda.synchronize()

    im = resize_image_for_wan_480p(load_image_from_file(start_image_path), sizes[0])
    print(im.height, im.width)
    
    #vae_size = module_size_mb(pipe.vae)
    #print(f"VAE total size: {vae_size:.2f} MB")
    
    #for idx, block in enumerate(pipe.transformer.blocks):
    #    print(f"block{idx:02d}: {module_size_mb(block):7.1f} MB")
            
    num_videos_per_prompt = 1
    max_sequence_length = 512
    prompt_embeds = None
    negative_prompt_embeds = None
    
    text_encoder_size = module_size_mb(pipe.text_encoder)
    print(f"Text Encoder total size: {text_encoder_size:.2f} MB")
    #tokenizer_size = module_size_mb(pipe.tokenizer)
    #print(f"Tokenizer total size: {tokenizer_size:.2f} MB")
    #print(pipe.tokenizer)                  #dont print this, it takes up so much space on the console
    
    pipe.set_progress_bar_config(desc="Pre-encoding text prompt")
    with pipe.progress_bar(total=1) as progress_bar:
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt                      = prompt1,
            negative_prompt             = negative_prompt,
            do_classifier_free_guidance = True,                      #throws an exception if you try to access the member variable for some reason
            num_videos_per_prompt       = num_videos_per_prompt,
            prompt_embeds               = None,
            negative_prompt_embeds      = None,
            max_sequence_length         = max_sequence_length,
            device                      = "cpu",
        )
        progress_bar.update()

    htmlCount = 0
    for seed in seeds:
      for guidance_scale in guidance_scales:
        for steps in steps_list:
            htmlCount += 1
            if htmlCount <= skipNum:
                continue
                    
            generator_states = []
            generator = torch.Generator(device="cpu").manual_seed(seed)
            
            frames = wani2v_custom_pipeline(
                    pipe,
                    image                  = im,
                    #prompt                = prompt1,
                    #negative_prompt       = negative_prompt,
                    prompt_embeds          = prompt_embeds,
                    negative_prompt_embeds = negative_prompt_embeds,
                    height                 = im.height,
                    width                  = im.width,
                    num_frames             = num_frames,
                    num_inference_steps    = steps,
                    generator              = generator,
                    guidance_scale         = guidance_scale,
                    num_videos_per_prompt  = num_videos_per_prompt,
                    max_sequence_length    = max_sequence_length,
                    use_sageattention      = use_sageattention
                ).frames
            output = frames[0]
            
            #This doesn't work
            #image64_frames = []
            #for i, frame in enumerate(frames):
            #    image64_frames.append(image_to_base64(Image.fromarray(frame)))
            
            #print(output)
            unix_time = int(time.time())
            
            if IS_WINDOWS:
                fileName = f"wan_gen\\{setName}\\{folderName}\\video_{unix_time}__{seed}_{htmlCount}.mp4"
            else:
                fileName = FILE_PREPEND / f"wan_gen/{setName}/{folderName}/video_{unix_time}__{seed}_{htmlCount}.mp4"
            
            print(fileName)
            ensure_folder_exists(fileName)
            export_to_video(output, fileName, fps=fps)
            
            pil_images = extract_frames(fileName)
            image64_frames = []
            for frame in pil_images:
                image64_frames.append(image_to_base64(frame))
            
            base64_video = mp4_to_base64(fileName)
                            
            generator_states.append(generator_to_b64(generator))
                            
            metadata = {
                "Prompt":            prompt1,
                "Negative Prompt":   negative_prompt,
                "Seed":              seed,
                "vid_width":         im.width,
                "vid_height":        im.height,
                "Steps":             steps,
                #"Strength":          inpainting_strength,
                "Guidance Scale":    guidance_scale,
                "LoRAs":             lora_string + " " + embedding_string,
                "Mask":              inpainting_mask_file,
                #"Model":             checkpoint_path,
                "Model":             REPO_ID_I2V,
                "Mode":              "Image2Video",
                "InputImage1":       start_image_path,
                "InputImage2":       "",
                "GeneratorStates":   generator_states,
                "FPS":               fps,
            }
            videos_data = [{
                "video_base64": base64_video,
                "image_frames": image64_frames,
                "metadata":     metadata
            }]
            
            html_content = generate_video_html(videos_data, python_script, job_script)
            fileName = os.path.splitext(fileName)[0] + ".html"
            save_html_file(html_content, fileName)
            print(f"[VIDEO] {htmlCount}: HTML file saved as {fileName}")

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    main()
