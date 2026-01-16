import torch
import gradio as gr
from diffusers import Flux2KleinPipeline
from PIL import Image
import numpy as np
import os
import random
from datetime import datetime
import uuid

# Configure Hugging Face Hub for better download performance and longer timeouts
# Set timeout to 10 minutes (600 seconds) for large model downloads
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
# Set etag timeout separately (this is what causes the 10 second timeout errors)
os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"
# Enable faster downloads with hf_xet if available
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Global variables for the pipelines
pipe_4b = None
pipe_4b_nvfp4 = None
pipe_4b_fp8 = None
pipe_9b = None
pipe_9b_nvfp4 = None
pipe_9b_fp8 = None
current_model = None
hf_token = None

# Default settings per mode
DEFAULT_STEPS = {
    "Distilled (4 steps)": 4,
    "Base (50 steps)": 50,
}

DEFAULT_GUIDANCE = {
    "Distilled (4 steps)": 3.5,
    "Base (50 steps)": 4.0,
}

MAX_SEED = np.iinfo(np.int32).max

# Create output directory for saved images
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_image_with_metadata(image, prompt, model_name, mode_name, width, height, steps, guidance, seed):
    """Save image with unique filename and metadata"""
    try:
        # Generate unique filename with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"flux2_klein_{timestamp}_{unique_id}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save image in PNG format (lossless, high quality)
        image.save(filepath, format="PNG", optimize=True)
        
        # Also save metadata as text file
        metadata_filename = f"flux2_klein_{timestamp}_{unique_id}_metadata.txt"
        metadata_filepath = os.path.join(OUTPUT_DIR, metadata_filename)
        
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            f.write(f"FLUX.2 [klein] Generation Metadata\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Mode: {mode_name}\n")
            f.write(f"Dimensions: {width}x{height}\n")
            f.write(f"Inference Steps: {steps}\n")
            f.write(f"Guidance Scale: {guidance}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def set_hf_token(token):
    """Set the Hugging Face token for model loading"""
    global hf_token
    if token and token.strip():
        hf_token = token.strip()
        return "✓ Token set successfully! You can now select a model and generate images."
    else:
        hf_token = None
        return "⚠ Token cleared or invalid."

def update_dimensions_from_image(image_list):
    """Update width/height sliders based on uploaded image aspect ratio.
    Keeps one side at 1024 and scales the other proportionally, with both sides as multiples of 8."""
    if image_list is None or len(image_list) == 0:
        return 1024, 1024  # Default dimensions
    
    # Get the first image to determine dimensions
    img = image_list[0][0] if isinstance(image_list[0], tuple) else image_list[0]
    img_width, img_height = img.size
    
    aspect_ratio = img_width / img_height
    
    if aspect_ratio >= 1:  # Landscape or square
        new_width = 1024
        new_height = int(1024 / aspect_ratio)
    else:  # Portrait
        new_height = 1024
        new_width = int(1024 * aspect_ratio)
    
    # Round to nearest multiple of 8
    new_width = round(new_width / 8) * 8
    new_height = round(new_height / 8) * 8
    
    # Ensure within valid range (minimum 256, maximum 1024)
    new_width = max(256, min(1024, new_width))
    new_height = max(256, min(1024, new_height))
    
    return new_width, new_height

def update_steps_from_mode(mode_choice):
    """Update the number of inference steps and guidance scale based on the selected mode."""
    return DEFAULT_STEPS[mode_choice], DEFAULT_GUIDANCE[mode_choice]

def apply_preset_size(preset):
    """Apply preset dimensions based on selected aspect ratio."""
    presets = {
        "Square (1:1) - 1024x1024": (1024, 1024),
        "Landscape (4:3) - 1024x768": (1024, 768),
        "Landscape (16:9) - 1024x576": (1024, 576),
        "Landscape (21:9) - 1344x576": (1344, 576),
        "Portrait (3:4) - 768x1024": (768, 1024),
        "Portrait (9:16) - 576x1024": (576, 1024),
        "Portrait (9:21) - 576x1344": (576, 1344),
        "Widescreen (16:9) - 1280x720": (1280, 720),
        "Widescreen (21:9) - 1536x640": (1536, 640),
    }
    if preset in presets:
        return presets[preset]
    return 1024, 1024  # Default

def load_model(model_choice):
    """Load the FLUX.2 klein model (all variants)"""
    global pipe_4b, pipe_4b_nvfp4, pipe_4b_fp8, pipe_9b, pipe_9b_nvfp4, pipe_9b_fp8, current_model, hf_token
    
    if hf_token is None:
        raise ValueError("Please set your Hugging Face token first!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # 4B Models
    if "4B (~13GB" in model_choice or model_choice == "FLUX.2 klein 4B (~13GB VRAM)":
        if pipe_4b is None:
            print("Loading FLUX.2 klein 4B model...")
            pipe_4b = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-4B", 
                torch_dtype=dtype,
                token=hf_token
            )
            if torch.cuda.is_available():
                pipe_4b.enable_model_cpu_offload()
            else:
                pipe_4b = pipe_4b.to(device)
            print("FLUX.2 klein 4B model loaded successfully!")
        current_model = pipe_4b
        return pipe_4b
    
    elif "4B FP8" in model_choice:
        if pipe_4b_fp8 is None:
            print("Loading FLUX.2 klein 4B FP8 (quantized) model...")
            pipe_4b_fp8 = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-4b-fp8", 
                torch_dtype=dtype,
                token=hf_token
            )
            if torch.cuda.is_available():
                pipe_4b_fp8.enable_model_cpu_offload()
            else:
                pipe_4b_fp8 = pipe_4b_fp8.to(device)
            print("FLUX.2 klein 4B FP8 model loaded successfully!")
        current_model = pipe_4b_fp8
        return pipe_4b_fp8
    
    elif "4B NVFP4" in model_choice:
        if pipe_4b_nvfp4 is None:
            print("Loading FLUX.2 klein 4B NVFP4 (quantized) model...")
            pipe_4b_nvfp4 = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-4b-nvfp4", 
                torch_dtype=dtype,
                token=hf_token
            )
            if torch.cuda.is_available():
                pipe_4b_nvfp4.enable_model_cpu_offload()
            else:
                pipe_4b_nvfp4 = pipe_4b_nvfp4.to(device)
            print("FLUX.2 klein 4B NVFP4 model loaded successfully!")
        current_model = pipe_4b_nvfp4
        return pipe_4b_nvfp4
    
    # 9B Models
    elif "9B (~29GB" in model_choice or model_choice == "FLUX.2 klein 9B (~29GB VRAM)":
        if pipe_9b is None:
            print("Loading FLUX.2 klein 9B model...")
            pipe_9b = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9B", 
                torch_dtype=dtype,
                token=hf_token
            )
            if torch.cuda.is_available():
                pipe_9b.enable_model_cpu_offload()
            else:
                pipe_9b = pipe_9b.to(device)
            print("FLUX.2 klein 9B model loaded successfully!")
        current_model = pipe_9b
        return pipe_9b
    
    elif "9B FP8" in model_choice:
        if pipe_9b_fp8 is None:
            print("Loading FLUX.2 klein 9B FP8 (quantized) model...")
            pipe_9b_fp8 = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9b-fp8", 
                torch_dtype=dtype,
                token=hf_token
            )
            if torch.cuda.is_available():
                pipe_9b_fp8.enable_model_cpu_offload()
            else:
                pipe_9b_fp8 = pipe_9b_fp8.to(device)
            print("FLUX.2 klein 9B FP8 model loaded successfully!")
        current_model = pipe_9b_fp8
        return pipe_9b_fp8
    
    else:  # 9B NVFP4
        if pipe_9b_nvfp4 is None:
            print("Loading FLUX.2 klein 9B NVFP4 (quantized) model...")
            pipe_9b_nvfp4 = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9b-nvfp4", 
                torch_dtype=dtype,
                token=hf_token
            )
            if torch.cuda.is_available():
                pipe_9b_nvfp4.enable_model_cpu_offload()
            else:
                pipe_9b_nvfp4 = pipe_9b_nvfp4.to(device)
            print("FLUX.2 klein 9B NVFP4 model loaded successfully!")
        current_model = pipe_9b_nvfp4
        return pipe_9b_nvfp4

def generate_image(
    prompt,
    input_images,
    mode_choice,
    model_choice,
    height,
    width,
    guidance_scale,
    num_inference_steps,
    seed,
    use_random_seed
):
    """Generate image from text prompt with optional input images"""
    try:
        # Check if token is set
        if hf_token is None:
            blank_image = Image.new('RGB', (512, 512), color='gray')
            return blank_image, "⚠ Error: Please set your Hugging Face token first!", seed
        
        # Load model if not already loaded
        pipeline = load_model(model_choice)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Handle seed (use int32 max to avoid overflow warnings)
        if use_random_seed:
            seed = random.randint(0, MAX_SEED)
        
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
        # Prepare image list (convert None or empty gallery to None)
        image_list = None
        if input_images is not None and len(input_images) > 0:
            image_list = []
            for item in input_images:
                # Handle both tuple (image, caption) and direct image
                img = item[0] if isinstance(item, tuple) else item
                image_list.append(img)
        
        # Generate image
        if "4B NVFP4" in model_choice:
            model_name = "4B NVFP4"
        elif "4B FP8" in model_choice:
            model_name = "4B FP8"
        elif "4B" in model_choice:
            model_name = "4B"
        elif "9B NVFP4" in model_choice:
            model_name = "9B NVFP4"
        elif "9B FP8" in model_choice:
            model_name = "9B FP8"
        else:
            model_name = "9B"
        
        mode_name = "Distilled" if "Distilled" in mode_choice else "Base"
        image_mode = " (with input images)" if image_list else ""
        print(f"Generating image with FLUX.2 klein {model_name} [{mode_name}]{image_mode}")
        print(f"Prompt: '{prompt}'")
        print(f"Parameters: {width}x{height}, steps: {num_inference_steps}, guidance: {guidance_scale}, seed: {seed}")
        
        # Build pipeline arguments
        pipe_kwargs = {
            "prompt": prompt,
            "height": int(height),
            "width": int(width),
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
            "generator": generator,
        }
        
        # Add images if provided
        if image_list is not None:
            pipe_kwargs["image"] = image_list
        
        result = pipeline(**pipe_kwargs)
        
        image = result.images[0]
        
        # Save image with unique filename
        saved_path = save_image_with_metadata(
            image, prompt, model_name, mode_name, 
            int(width), int(height), int(num_inference_steps), 
            float(guidance_scale), seed
        )
        
        if saved_path:
            status_msg = f"✓ Generated successfully with {model_name} model [{mode_name}]!\n📁 Saved as: {os.path.basename(saved_path)}\n🌱 Seed: {seed}"
        else:
            status_msg = f"✓ Generated successfully with {model_name} model [{mode_name}]! (Warning: Could not save to disk)\n🌱 Seed: {seed}"
        
        return image, status_msg, seed
    
    except ValueError as e:
        error_msg = f"⚠ {str(e)}"
        print(error_msg)
        blank_image = Image.new('RGB', (512, 512), color='gray')
        return blank_image, error_msg, seed
    except Exception as e:
        error_msg = str(e)
        # Check for common authentication errors
        if "401" in error_msg or "authentication" in error_msg.lower() or "access" in error_msg.lower():
            error_msg = "⚠ Authentication Error: Invalid token or you haven't accepted the model license. Please check your token and make sure you've accepted the license on Hugging Face."
        else:
            error_msg = f"⚠ Error generating image: {error_msg}"
        print(error_msg)
        # Return a blank image and error message
        blank_image = Image.new('RGB', (512, 512), color='gray')
        return blank_image, error_msg, seed

# CSS for better styling
css = """
#col-container {
    margin: 0 auto;
    max-width: 1400px;
}
.gallery-container img {
    object-fit: contain;
}
"""

# Create Gradio interface
with gr.Blocks(title="FLUX.2 [klein] Image Generator", css=css) as demo:
    with gr.Column(elem_id="col-container"):
    gr.Markdown(f"""
    # 🎨 FLUX.2 [klein] Image Generator
    
    Generate high-quality images from text descriptions using Black Forest Labs' FLUX.2 [klein] models.
    
    **Features:**
    - ⚡ Sub-second image generation with outstanding quality
    - 🎯 Step-distilled to 4 inference steps for optimal speed
    - 🗜️ Multiple quantization options (Base/FP8/NVFP4) for different VRAM requirements
    - 💻 Choose between 4B (consumer GPUs) or 9B (high-end GPUs) models
    - 💾 **Auto-save**: All images saved to `{OUTPUT_DIR}/` in PNG format with metadata
    
    *Note: First generation will take longer as the model loads into memory.*
    """)
    
    # Hugging Face Token Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ### 🔑 Authentication Required
            These models are gated. You need a Hugging Face token with access to the FLUX.2 [klein] models.
            
            **Steps:**
            1. Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
            2. Accept the license at [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and/or [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
            3. Enter your token below
            """)
            
            with gr.Row():
                token_input = gr.Textbox(
                    label="Hugging Face Token",
                    placeholder="hf_...",
                    type="password",
                    scale=4
                )
                token_btn = gr.Button("Set Token", variant="primary", scale=1)
            
            token_status = gr.Textbox(
                label="Status",
                interactive=False,
                value="⚠ Please set your Hugging Face token to continue"
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="A cat holding a sign that says hello world",
                lines=3,
                value="A cat holding a sign that says hello world"
            )
            
            with gr.Accordion("Input Image(s) (optional)", open=False):
                input_images = gr.Gallery(
                    label="Input Image(s) for editing/combining",
                    type="pil",
                    columns=3,
                    rows=1,
                    show_label=False
                )
            
            mode_choice = gr.Radio(
                label="Mode",
                choices=["Distilled (4 steps)", "Base (50 steps)"],
                value="Distilled (4 steps)",
                info="Distilled = fast (4 steps), Base = traditional (50 steps)"
            )
            
            model_selector = gr.Radio(
                choices=[
                    "FLUX.2 klein 4B (~13GB VRAM)",
                    "FLUX.2 klein 4B FP8 (~7GB VRAM - Quantized)",
                    "FLUX.2 klein 4B NVFP4 (~4GB VRAM - Quantized)",
                    "FLUX.2 klein 9B (~29GB VRAM)",
                    "FLUX.2 klein 9B FP8 (~15GB VRAM - Quantized)",
                    "FLUX.2 klein 9B NVFP4 (~8GB VRAM - Quantized)"
                ],
                value="FLUX.2 klein 4B FP8 (~7GB VRAM - Quantized)",
                label="Model Selection",
                info="4B = consumer GPUs, 9B = high-end GPUs | Base = full precision, FP8 = best balance, NVFP4 = lowest VRAM"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                size_preset = gr.Dropdown(
                    label="Size Preset",
                    choices=[
                        "Square (1:1) - 1024x1024",
                        "Landscape (4:3) - 1024x768",
                        "Landscape (16:9) - 1024x576",
                        "Landscape (21:9) - 1344x576",
                        "Portrait (3:4) - 768x1024",
                        "Portrait (9:16) - 576x1024",
                        "Portrait (9:21) - 576x1344",
                        "Widescreen (16:9) - 1280x720",
                        "Widescreen (21:9) - 1536x640",
                    ],
                    value="Square (1:1) - 1024x1024",
                    info="Select a preset or manually adjust sliders below"
                )
                
                with gr.Row():
                    width_slider = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        step=8,
                        value=1024,
                        label="Width"
                    )
                    height_slider = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        step=8,
                        value=1024,
                        label="Height"
                    )
                
                guidance_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=3.5,
                    label="Guidance Scale",
                    info="Higher values follow the prompt more closely"
                )
                
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=4,
                    label="Inference Steps",
                    info="Auto-updates based on mode. Distilled optimized for 4 steps."
                )
                
                use_random_seed = gr.Checkbox(
                    label="Use Random Seed",
                    value=True
                )
                
                seed_input = gr.Number(
                    label="Seed (ignored if random seed is enabled)",
                    value=0,
                    precision=0
                )
            
            generate_btn = gr.Button("🚀 Generate Image", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=600
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False
            )
            seed_output = gr.Number(
                label="Seed Used",
                value=0,
                interactive=False,
                visible=False
            )
    
        # Example prompts
        gr.Examples(
            examples=[
                ["A cat holding a sign that says hello world"],
                ["A futuristic cityscape at sunset with flying cars, neon lights, cyberpunk style"],
                ["A photorealistic portrait of a robot with human emotions, detailed metal textures"],
                ["An enchanted forest with glowing mushrooms and fairy lights, magical atmosphere"],
                ["A steaming cup of coffee on a wooden table, morning light, cozy atmosphere"],
                ["A majestic dragon perched on a mountain peak, detailed scales, dramatic sky"],
            ],
            inputs=[prompt_input],
            label="Example Prompts"
        )
        
        gr.Markdown("""
        ---
        ### 📚 Understanding Model Options
        
        #### 🎯 Mode (Inference Steps)
        - **Distilled (4 steps)**: Fast mode optimized for sub-second generation - **RECOMMENDED**
        - **Base (50 steps)**: Traditional diffusion with more refinement passes
        - The models are specifically trained for 4 steps - use Distilled mode!
        
        #### 🗜️ Quantization (Memory Precision)
        - **Base (BF16)**: Full 16-bit precision - highest quality, most VRAM (~13GB for 4B, ~29GB for 9B)
        - **FP8**: 8-bit precision - ~50% less VRAM, minimal quality loss - **BEST BALANCE**
        - **NVFP4**: 4-bit precision - ~75% less VRAM, slight quality trade-off - best for limited VRAM
        
        #### 📦 Model Size
        - **4B Models**: 4 billion parameters, consumer GPUs (RTX 3090/4070+), Apache 2.0 license
        - **9B Models**: 9 billion parameters, high-end GPUs (RTX 4090+), non-commercial license
        
        💡 **Pro Tip**: Mode, quantization, and model size are independent! You can use **9B NVFP4** (large model, low VRAM) with **Distilled mode** (4 steps, fast generation).
        
        ---
        ### 🖼️ Image Editing & Combining
        
        Upload 1 or more images to enable **image-to-image editing** and **multi-reference image combining**:
        - **Single image**: Edit or transform the image based on your prompt
        - **Multiple images**: Combine elements from different images (e.g., "Person from image 1 with background from image 2")
        - Dimensions auto-adjust to match your input image aspect ratio
        
        ---
        ### About FLUX.2 [klein]
        
        The FLUX.2 [klein] model family are Black Forest Labs' **fastest image models to date**. FLUX.2 [klein] unifies generation and editing in a single compact architecture, delivering **state-of-the-art quality with end-to-end inference in as low as under a second**. Built for applications that require real-time image generation without sacrificing quality.
        
        **Key Features:**
        - ⚡ Sub-second image generation with outstanding quality
        - 🎯 Defines the Pareto frontier for quality vs. latency
        - 💪 Matches or exceeds models 5x its size—in under half a second
        - 🔧 Text-to-image and image-to-image multi-reference editing in a unified model
        - 🎨 Excellent prompt adherence and output diversity
        - ⚙️ Step-distilled to 4 inference steps for optimal performance
        
        ### Model Information:
        
        **4B Models** (Consumer GPUs, Apache 2.0 ✓ Commercial Use)
        - **4B Base**: Full BF16 precision, ~13GB VRAM (RTX 3090/4070+)
        - **4B FP8**: 8-bit quantized, ~7GB VRAM (RTX 3070+) - **RECOMMENDED**
        - **4B NVFP4**: 4-bit quantized, ~4GB VRAM (RTX 3060+)
        
        **9B Models** (High-end GPUs, Non-Commercial License)
        - **9B Base**: Full BF16 precision, ~29GB VRAM (RTX 4090+)
        - **9B FP8**: 8-bit quantized, ~15GB VRAM (RTX 3090 Ti+) - **RECOMMENDED**
        - **9B NVFP4**: 4-bit quantized, ~8GB VRAM (RTX 3070+)
        
        All quantized variants maintain excellent quality with significantly reduced VRAM usage.
        
        ### Tips for better results:
        - Be specific and descriptive in your prompts
        - Use **Distilled mode (4 steps)** for best speed/quality balance
        - Use guidance scale around 3.5-4.5 for best results
        - **Use size presets** for common aspect ratios (square, landscape, portrait, widescreen)
        - Standard resolutions: 1024x1024, 1024x768, 768x1024, 1280x720, 1536x640
        - For image editing: Upload images and describe what you want to change
        
        ### Limitations:
        - Not intended to provide factual information
        - Text rendering may be inaccurate or distorted
        - May represent or amplify biases from training data
        - Prompt following is influenced by prompting style
        
        ### Responsible AI:
        Black Forest Labs is committed to responsible AI development. These models have undergone extensive safety evaluations and include built-in mitigations. For safety concerns: **safety@blackforestlabs.ai**
        
        ---
        
        **Learn more:** [Black Forest Labs Blog](https://blackforestlabs.ai/blog) | [Model Cards](https://huggingface.co/black-forest-labs)
        """)
    
    # Connect the token button
    token_btn.click(
        fn=set_hf_token,
        inputs=[token_input],
        outputs=[token_status]
    )
    
    # Auto-update dimensions when images are uploaded
    input_images.upload(
        fn=update_dimensions_from_image,
        inputs=[input_images],
        outputs=[width_slider, height_slider]
    )
    
    # Auto-update steps and guidance when mode changes
    mode_choice.change(
        fn=update_steps_from_mode,
        inputs=[mode_choice],
        outputs=[steps_slider, guidance_scale_slider]
    )
    
    # Auto-update dimensions when preset is selected
    size_preset.change(
        fn=apply_preset_size,
        inputs=[size_preset],
        outputs=[width_slider, height_slider]
    )
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            input_images,
            mode_choice,
            model_selector,
            height_slider,
            width_slider,
            guidance_scale_slider,
            steps_slider,
            seed_input,
            use_random_seed
        ],
        outputs=[output_image, status_text, seed_output]
    )

# Launch the app
if __name__ == "__main__":
    print("=" * 60)
    print("FLUX.2 [klein] Gradio UI")
    print("=" * 60)
    print("\nAvailable models:")
    print("  4B Models (Consumer GPUs, Apache 2.0 License):")
    print("    - 4B Base (~13GB VRAM) - Full precision")
    print("    - 4B FP8 (~7GB VRAM) - RECOMMENDED - Best balance")
    print("    - 4B NVFP4 (~4GB VRAM) - Lowest VRAM requirement")
    print("  9B Models (High-end GPUs, Non-Commercial License):")
    print("    - 9B Base (~29GB VRAM) - Highest quality")
    print("    - 9B FP8 (~15GB VRAM) - RECOMMENDED - Best balance")
    print("    - 9B NVFP4 (~8GB VRAM) - High quality on mid-range GPUs")
    print()
    print("💾 Images will be automatically saved to:")
    print(f"   {os.path.abspath(OUTPUT_DIR)}/")
    print("   Format: PNG (lossless) with metadata txt file")
    print()
    print("⚠ You will need to set your Hugging Face token in the web interface")
    print()
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"VRAM: {vram:.1f}GB")
        if vram >= 28:
            print("✓ Excellent! Can run all models including 9B Base")
            print("  Recommended: 9B FP8 for best balance")
        elif vram >= 14:
            print("✓ Great! Can run 9B FP8 (~15GB) or any 4B model")
            print("  Recommended: 9B FP8 for quality or 4B FP8 for speed")
        elif vram >= 12:
            print("✓ Good! Can run 4B Base (~13GB) or 4B FP8 (~7GB)")
            print("  Recommended: 4B FP8 for best balance")
        elif vram >= 6:
            print("✓ Sufficient for 4B FP8 (~7GB)")
            print("  Recommended: 4B FP8")
        elif vram >= 4:
            print("⚠ Limited VRAM - use 4B NVFP4 (~4GB)")
        else:
            print("⚠ Warning: Very limited VRAM. 4B NVFP4 requires ~4GB minimum")
    else:
        print("⚠ No CUDA GPU detected. Generation will be very slow on CPU.")
    
    print()
    print("Starting web interface...")
    print("=" * 60)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
