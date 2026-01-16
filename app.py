import torch
import gradio as gr
from diffusers import Flux2KleinPipeline
from PIL import Image
import numpy as np
import os

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

def set_hf_token(token):
    """Set the Hugging Face token for model loading"""
    global hf_token
    if token and token.strip():
        hf_token = token.strip()
        return "✓ Token set successfully! You can now select a model and generate images."
    else:
        hf_token = None
        return "⚠ Token cleared or invalid."

def load_model(model_choice):
    """Load the FLUX.2 klein model (all variants)"""
    global pipe_4b, pipe_4b_nvfp4, pipe_4b_fp8, pipe_9b, pipe_9b_nvfp4, pipe_9b_fp8, current_model, hf_token
    
    if hf_token is None:
        raise ValueError("Please set your Hugging Face token first!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # 4B Models
    if model_choice == "FLUX.2 klein 4B (~13GB VRAM)":
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
    
    elif model_choice == "FLUX.2 klein 4B NVFP4 (~13GB VRAM - Quantized)":
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
    
    elif model_choice == "FLUX.2 klein 4B FP8 (~13GB VRAM - Quantized)":
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
    
    # 9B Models
    elif model_choice == "FLUX.2 klein 9B (~29GB VRAM)":
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
    
    elif model_choice == "FLUX.2 klein 9B NVFP4 (~29GB VRAM - Quantized)":
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
    
    else:  # 9B FP8
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

def generate_image(
    prompt,
    model_choice,
    height,
    width,
    guidance_scale,
    num_inference_steps,
    seed,
    use_random_seed
):
    """Generate image from text prompt"""
    try:
        # Check if token is set
        if hf_token is None:
            blank_image = Image.new('RGB', (512, 512), color='gray')
            return blank_image, "⚠ Error: Please set your Hugging Face token first!"
        
        # Load model if not already loaded
        pipeline = load_model(model_choice)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Handle seed (use int32 max to avoid overflow warnings)
        if use_random_seed:
            seed = np.random.randint(0, 2**31 - 1)
        
        generator = torch.Generator(device=device).manual_seed(int(seed))
        
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
        print(f"Generating image with FLUX.2 klein {model_name} - prompt: '{prompt}'")
        print(f"Parameters: {width}x{height}, steps: {num_inference_steps}, guidance: {guidance_scale}, seed: {seed}")
        
        result = pipeline(
            prompt=prompt,
            height=int(height),
            width=int(width),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            generator=generator
        )
        
        image = result.images[0]
        
        return image, f"✓ Generated successfully with {model_name} model! Seed used: {seed}"
    
    except ValueError as e:
        error_msg = f"⚠ {str(e)}"
        print(error_msg)
        blank_image = Image.new('RGB', (512, 512), color='gray')
        return blank_image, error_msg
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
        return blank_image, error_msg

# Create Gradio interface
with gr.Blocks(title="FLUX.2 [klein] Image Generator") as demo:
    gr.Markdown("""
    # 🎨 FLUX.2 [klein] Image Generator
    
    Generate high-quality images from text descriptions using Black Forest Labs' FLUX.2 [klein] models.
    
    **Features:**
    - Sub-second image generation with outstanding quality
    - Excellent prompt adherence and output diversity
    - Real-time generation capabilities
    - Choose between 4B (consumer GPUs) or 9B (high-end GPUs) models
    
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
            
            model_selector = gr.Radio(
                choices=[
                    "FLUX.2 klein 4B (~13GB VRAM)",
                    "FLUX.2 klein 4B NVFP4 (~13GB VRAM - Quantized)",
                    "FLUX.2 klein 4B FP8 (~13GB VRAM - Quantized)",
                    "FLUX.2 klein 9B (~29GB VRAM)",
                    "FLUX.2 klein 9B NVFP4 (~29GB VRAM - Quantized)",
                    "FLUX.2 klein 9B FP8 (~29GB VRAM - Quantized)"
                ],
                value="FLUX.2 klein 4B (~13GB VRAM)",
                label="Model Selection",
                info="Choose base model (4B/9B) or quantized variants (NVFP4/FP8) for optimized performance"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    width_slider = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1024,
                        label="Width"
                    )
                    height_slider = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1024,
                        label="Height"
                    )
                
                guidance_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.5,
                    value=4.0,
                    label="Guidance Scale",
                    info="Higher values follow the prompt more closely"
                )
                
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                    label="Inference Steps",
                    info="More steps = better quality but slower. Model is optimized for 4 steps."
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
    
    # Example prompts
    gr.Examples(
        examples=[
            ["A cat holding a sign that says hello world", 1024, 1024, 4.0, 4],
            ["A futuristic cityscape at sunset with flying cars", 1024, 1024, 4.0, 4],
            ["A photorealistic portrait of a robot with human emotions", 1024, 1024, 4.0, 4],
            ["An enchanted forest with glowing mushrooms and fairy lights", 1024, 1024, 4.0, 4],
            ["A steaming cup of coffee on a wooden table, morning light, cozy atmosphere", 1024, 1024, 4.0, 4],
            ["A majestic dragon perched on a mountain peak, detailed scales, dramatic sky", 1024, 1024, 4.0, 4],
        ],
        inputs=[prompt_input, width_slider, height_slider, guidance_scale_slider, steps_slider],
        label="Example Prompts"
    )
    
    gr.Markdown("""
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
    
    **4B Models** (Consumer GPUs: RTX 3090/4070+, ~13GB VRAM, Apache 2.0 ✓)
    - **4B Base**: Full precision, 4 billion parameters
    - **4B NVFP4**: NVFP4 quantized for optimized performance
    - **4B FP8**: FP8 quantized for maximum efficiency
    
    **9B Models** (High-end GPUs: RTX 4090+, ~29GB VRAM, Non-Commercial License)
    - **9B Base**: Full BF16 precision, 9B flow + 8B Qwen3 text embedder
    - **9B NVFP4**: NVFP4 quantized for optimized inference
    - **9B FP8**: FP8 quantized for faster generation
    
    All quantized variants offer similar quality to base models with improved performance and memory efficiency.
    
    ### Tips for better results:
    - Be specific and descriptive in your prompts
    - The models are optimized for 4 inference steps
    - Use guidance scale around 3.5-4.5 for best results
    - Standard resolutions: 1024x1024, 1024x768, 768x1024
    
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
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            model_selector,
            height_slider,
            width_slider,
            guidance_scale_slider,
            steps_slider,
            seed_input,
            use_random_seed
        ],
        outputs=[output_image, status_text]
    )

# Launch the app
if __name__ == "__main__":
    print("=" * 60)
    print("FLUX.2 [klein] Gradio UI")
    print("=" * 60)
    print("\nAvailable models:")
    print("  4B Models (Consumer GPUs: RTX 3090/4070+, ~13GB VRAM):")
    print("    - 4B (Full Precision)")
    print("    - 4B NVFP4 (Quantized)")
    print("    - 4B FP8 (Quantized)")
    print("  9B Models (High-end GPUs: RTX 4090+, ~29GB VRAM):")
    print("    - 9B (Full Precision)")
    print("    - 9B NVFP4 (Quantized)")
    print("    - 9B FP8 (Quantized)")
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
            print("✓ Sufficient VRAM for both 4B and 9B models")
        elif vram >= 12:
            print("✓ Sufficient VRAM for 4B model (~13GB required)")
        else:
            print("⚠ Warning: May not have sufficient VRAM. 4B model requires ~13GB")
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
