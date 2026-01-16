# FLUX.2 [klein] - Gradio UI

<div align="center">

🎨 **Fast Text-to-Image Generation with Black Forest Labs' FLUX.2 Models**

[![License](https://img.shields.io/badge/License-Mixed-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.50-orange.svg)](https://gradio.app/)

</div>

---

## 🌟 Overview

A beautiful Gradio web interface for **FLUX.2 [klein]** models from Black Forest Labs. Generate stunning images from text descriptions in under a second with state-of-the-art quality. The FLUX.2 [klein] model family are Black Forest Labs' **fastest image models to date**, unifying generation and editing in a single compact architecture with **end-to-end inference in as low as under a second**. Choose from **six model variants** (4B and 9B base models plus NVFP4 and FP8 quantized versions) optimized for different hardware configurations and use cases.

### ✨ Key Features

- ⚡ **Sub-second image generation** with outstanding quality
- 🎯 **Pareto frontier performance** - matches or exceeds models 5x its size in under half a second
- 🔄 **Unified architecture** - text-to-image and image-to-image multi-reference editing in one model
- 💻 **Consumer GPU support** (RTX 3090/4070+)
- 🎛️ **Six model variants**: 4B, 4B NVFP4, 4B FP8, 9B, 9B NVFP4, 9B FP8
- 🖼️ **Flexible resolution** support (256x256 to 2048x2048)
- ⚙️ **Optimized for speed** - step-distilled to 4 inference steps
- 🎨 **Excellent prompt adherence** and creative exploration
- 🔧 **Advanced controls**: guidance scale, inference steps, seed control
- 💾 **Auto-save images**: All generated images automatically saved to `outputs/` in PNG format with metadata
- 📝 **Example prompts** included to get you started

---

## 🤖 Model Information

### 4B Models (Consumer GPUs: RTX 3090/4070+, ~13GB VRAM)

#### FLUX.2 [klein] 4B - Full Precision
- **Size**: 4 billion parameters (full precision)
- **License**: Apache 2.0 (✓ commercial use allowed)
- **Best for**: Production use, commercial applications
- **Performance**: Fast generation on consumer GPUs
- **Repository**: [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

#### FLUX.2 [klein] 4B NVFP4 - Quantized
- **Size**: 4 billion parameters (NVFP4 quantized)
- **License**: Apache 2.0 (✓ commercial use allowed)
- **Best for**: Optimized performance, edge deployment
- **Performance**: Faster inference with minimal quality loss
- **Repository**: [FLUX.2-klein-4b-nvfp4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4)

#### FLUX.2 [klein] 4B FP8 - Quantized
- **Size**: 4 billion parameters (FP8 quantized)
- **License**: Apache 2.0 (✓ commercial use allowed)
- **Best for**: Maximum efficiency, resource-constrained environments
- **Performance**: Excellent speed/quality balance
- **Repository**: [FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8)

### 9B Models (High-end GPUs: RTX 4090+, ~29GB VRAM)

#### FLUX.2 [klein] 9B - Full Precision
- **Size**: 9 billion parameters (9B flow model + 8B Qwen3 text embedder)
- **Architecture**: Rectified flow transformer (full BF16 weights)
- **License**: Non-Commercial License
- **Best for**: Highest quality, research applications
- **Performance**: State-of-the-art quality, matches models 5x its size
- **Repository**: [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)

#### FLUX.2 [klein] 9B NVFP4 - Quantized
- **Size**: 9 billion parameters (NVFP4 quantized)
- **License**: Non-Commercial License
- **Best for**: Performance-optimized high-quality inference
- **Performance**: Faster than full precision with excellent quality
- **Repository**: [FLUX.2-klein-9b-nvfp4](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4)

#### FLUX.2 [klein] 9B FP8 - Quantized
- **Size**: 9 billion parameters (FP8 quantized)
- **License**: Non-Commercial License
- **Best for**: Balanced high-quality and performance
- **Performance**: Best speed/quality balance for 9B
- **Repository**: [FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8)

All models support:
- Text-to-image generation
- Image-to-image multi-reference editing
- Real-time generation and application integration
- Step-distilled to 4 inference steps for optimal speed/quality balance

### Understanding Model Concepts

#### 🎯 Model Distillation (Inference Speed)
FLUX.2 [klein] is **step-distilled** to 4 inference steps for optimal generation speed:
- Sub-second generation with optimized performance
- The models are specifically trained for 4 steps
- No need to adjust - default settings provide the best results

#### 🗜️ Model Quantization (Memory Precision)
Reduces VRAM usage and increases speed by using lower-precision numbers:

**Full Precision (Base Models)**
- BF16 (16-bit) floating point weights
- Maximum quality, highest VRAM usage
- Example: 9B model uses ~29GB VRAM
- Best when quality is absolute priority

**FP8 Quantization**
- 8-bit floating point format
- ~50% memory reduction vs BF16
- Minimal quality loss
- Example: 9B FP8 uses ~15GB VRAM
- **Recommended** for best balance

**NVFP4 Quantization**
- NVIDIA 4-bit floating point format
- ~75% memory reduction vs BF16
- Slight quality trade-off for maximum efficiency
- Example: 9B NVFP4 uses ~8GB VRAM
- Best for resource-constrained GPUs

#### 🔄 How They Combine
These are **independent concepts** you can choose:
- **Model Size**: 4B vs 9B (parameter count)
- **Quantization**: Base vs FP8 vs NVFP4 (memory precision)
- **Inference Steps**: Optimized for 4 steps (distilled for fast generation)

Example: You can use **9B NVFP4** (large model, low VRAM) for high-quality sub-second generation

---

## 💾 Hardware Requirements

### Budget GPUs (4B NVFP4)
- **GPU**: NVIDIA RTX 3060 (12GB) or better
- **VRAM**: 4GB+
- **RAM**: 16GB+
- **Storage**: 15GB+ free space
- **Best for**: Entry-level GPUs, maximum efficiency

### Consumer GPUs (4B / 4B FP8)
- **GPU**: NVIDIA RTX 3070/3090 / RTX 4070 or better
- **VRAM**: 7-13GB
- **RAM**: 16GB+
- **Storage**: 20GB+ free space
- **Best for**: Most users, great quality/performance balance

### High-end GPUs (9B FP8)
- **GPU**: NVIDIA RTX 3090 Ti / RTX 4080 or better
- **VRAM**: 15GB+
- **RAM**: 32GB+
- **Storage**: 35GB+ free space
- **Best for**: Better quality, still accessible

### Enthusiast GPUs (9B Base)
- **GPU**: NVIDIA RTX 4090 or better
- **VRAM**: 29GB+
- **RAM**: 32GB+
- **Storage**: 40GB+ free space
- **Best for**: Maximum quality, research

### Software
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.12+
- **CUDA**: 11.8+ (for NVIDIA GPUs)
- **Drivers**: Latest NVIDIA drivers recommended

---

## 📦 Installation

### Option 1: Pinokio (Easiest)

1. Install [Pinokio](https://pinokio.computer/)
2. Search for "FLUX.2 klein" in Pinokio
3. Click "Install"
4. Click "Start" when installation completes

### Option 2: Manual Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flux-2-klein-pinokio.git
   cd flux-2-klein-pinokio
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   
   # Windows
   env\Scripts\activate
   
   # Linux/Mac
   source env/bin/activate
   ```

3. **Install PyTorch**
   ```bash
   # NVIDIA GPU (CUDA)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   
   # CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser** to `http://localhost:7860`

---

## 🔑 Hugging Face Token Setup

These models are **gated** and require authentication.

### Steps:

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co/)

2. **Get your access token**
   - Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token (read access is sufficient)

3. **Accept the model licenses**
   - [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
   - [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)

4. **Enter your token in the web interface**
   - Open the web UI
   - Enter your token in the "Hugging Face Token" field
   - Click "Set Token"
   - Start generating!

**Note**: Your token is stored in memory for the current session only. You'll need to re-enter it each time you restart the application.

---

## 🎨 Usage

### Basic Generation

1. **Set your Hugging Face token** (required on first use)
2. **Select a model**: Choose from 6 variants based on your GPU VRAM
   - **4B Models** (Consumer GPUs, Apache 2.0 License):
     - 4B Base (~13GB VRAM) - Full precision
     - 4B FP8 (~7GB VRAM) - **RECOMMENDED** - Best balance
     - 4B NVFP4 (~4GB VRAM) - Lowest VRAM
   - **9B Models** (High-end GPUs, Non-Commercial License):
     - 9B Base (~29GB VRAM) - Highest quality
     - 9B FP8 (~15GB VRAM) - **RECOMMENDED** - Best balance
     - 9B NVFP4 (~8GB VRAM) - Accessible high quality
3. **Enter a prompt**: Be specific and descriptive
4. **Click "Generate Image"**
5. **Wait a few seconds** (first generation loads the model - 30-60 seconds)
6. **Enjoy your image!**

💡 **First time users**: Start with **4B FP8** - it works on most GPUs and offers excellent quality!

### Output Files

All generated images are automatically saved to the `outputs/` directory:
- **Image files**: PNG format (lossless, high quality) with unique filenames
- **Metadata files**: Accompanying .txt files with generation parameters
- **Filename format**: `flux2_klein_YYYYMMDD_HHMMSS_[uuid].png`

Example:
```
outputs/
  ├── flux2_klein_20260116_143052_a3b9c1d2.png
  ├── flux2_klein_20260116_143052_a3b9c1d2_metadata.txt
  ├── flux2_klein_20260116_144328_f7e2a8b4.png
  └── flux2_klein_20260116_144328_f7e2a8b4_metadata.txt
```

The metadata file contains:
- Prompt used
- Model and mode selected
- Image dimensions
- Inference steps and guidance scale
- Seed value (for reproducibility)
- Generation timestamp

### Advanced Settings

<details>
<summary>Click to expand</summary>

#### Resolution
- **Size Presets**: Quick selection for common aspect ratios
  - Square: 1024x1024
  - Landscape: 1024x768, 1024x576, 1344x576, 1280x720, 1536x640
  - Portrait: 768x1024, 576x1024, 576x1344
- **Custom**: Manually adjust width/height (256-2048 pixels, steps of 8)
- **Recommended**: Use presets for best results

#### Guidance Scale
- **Range**: 1.0-10.0
- **Recommended**: 3.5-4.5
- **Effect**: Higher values = stronger prompt adherence

#### Inference Steps
- **Default**: 4 steps (models are step-distilled for this)
- **Range**: 1-100 steps available (advanced users only)
- **Recommended**: Keep at 4 steps for optimal performance
- **Note**: More steps may not improve quality as models are trained for 4-step generation

#### Seed Control
- **Random Seed**: Different result each time
- **Fixed Seed**: Reproducible results

</details>

### Example Prompts

Try these to get started:

```
A cat holding a sign that says hello world
```

```
A futuristic cityscape at sunset with flying cars, neon lights, cyberpunk style
```

```
A photorealistic portrait of a robot with human emotions, detailed metal textures
```

```
An enchanted forest with glowing mushrooms and fairy lights, magical atmosphere
```

```
A steaming cup of coffee on a wooden table, morning light, cozy atmosphere
```

---

## 💡 Tips for Better Results

### Prompting Tips
1. **Be specific and descriptive** in your prompts
2. **Include style keywords**: "photorealistic", "oil painting", "digital art"
3. **Add lighting details**: "golden hour", "studio lighting", "dramatic shadows"
4. **Specify composition**: "close-up", "wide angle", "aerial view"

### Model Selection Tips
5. **Start with 4B FP8** - best balance for testing (works on most GPUs)
6. **Use FP8 variants** for optimal speed/quality/VRAM balance - **RECOMMENDED**
7. **Try NVFP4 for limited VRAM** - 75% memory savings with good quality
8. **Upgrade to 9B FP8** if you have 16GB+ VRAM for better quality

### Generation Settings
9. **Use default 4 inference steps** - models are specifically optimized for this
10. **Keep guidance scale around 3.5-4.5** for best results
11. **Use size presets** for quick resolution selection
12. **Standard resolutions**: 1024x1024, 1024x768, 768x1024, 1280x720, 1536x640

💡 **Pro Tip**: Quantization (FP8/NVFP4) lets you use larger models on lower-VRAM GPUs! Choose **9B NVFP4** for high quality on mid-range GPUs, or **4B FP8** for best balance on consumer hardware.

---

## ⚠️ Limitations

- **Not for factual information**: Models are not designed to provide accurate factual content
- **Text rendering**: Generated text may be inaccurate or distorted
- **Biases**: As statistical models, they may represent or amplify biases from training data
- **Prompt matching**: Models may not always generate outputs that perfectly match prompts
- **Style sensitivity**: Prompt following is heavily influenced by prompting style

---

## 🛡️ Responsible AI & Safety

Black Forest Labs is committed to responsible AI development. The FLUX.2 [klein] models have undergone:

- **Pre-training mitigation**: NSFW and CSAM filtering in training data (partnership with [IWF](https://www.iwf.org.uk/))
- **Post-training mitigation**: Multiple rounds of safety fine-tuning against T2I and I2I abuse
- **Extensive evaluation**: Internal and third-party adversarial testing
- **Safety filters**: Built-in content filtering (NSFW, protected content)
- **Content provenance**: Pixel-layer watermarking and C2PA metadata support

### Out-of-Scope Use

These models **may not** be used for:
- Exploiting or harming minors in any way
- Generating deceptive, fraudulent, or harmful content
- Creating non-consensual intimate imagery or illegal content
- Harassment, abuse, threats, or bullying
- High-risk automated decision making affecting legal rights

**Safety Contact**: safety@blackforestlabs.ai

For full details, see the [model cards on Hugging Face](https://huggingface.co/black-forest-labs).

---

## 🗂️ Project Structure

```
flux-2-klein-pinokio/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── pinokio.js            # Pinokio configuration
├── install.js            # Installation script
├── start.js              # Startup script
├── update.js             # Update script
├── reset.js              # Reset script
├── link.js               # Deduplication script
├── torch.js              # PyTorch installer
├── icon.png              # Application icon
└── README.md             # This file
```

---

## 🔧 Troubleshooting

### "Please set your Hugging Face token"
- Enter your HF token in the web interface
- Make sure you've accepted the model licenses on Hugging Face

### "Authentication Error"
- Verify your token is correct
- Check that you've accepted the model licenses
- Try creating a new token with read permissions

### "Out of memory" errors
- Close other applications to free VRAM
- Use the 4B model instead of 9B
- Reduce resolution (e.g., 512x512)
- Restart the application

### Slow generation
- First generation loads the model (30-60 seconds)
- Subsequent generations are much faster
- Ensure you're using a CUDA-compatible GPU
- Check GPU utilization with `nvidia-smi`

### Model won't load
- Check your internet connection (models download from HF)
- Verify you have enough free disk space
- Try clearing your HuggingFace cache: `~/.cache/huggingface/`

---

## 📝 License

### This Project
This Gradio interface is released under the **MIT License**.

### Model Licenses

**4B Models (4B, 4B NVFP4, 4B FP8)**
- **License**: Apache 2.0
- **Usage**: ✓ Commercial use allowed
- **Best for**: Production applications, commercial products, edge deployment

**9B Models (9B, 9B NVFP4, 9B FP8)**
- **License**: FLUX Non-Commercial License
- **Usage**: Non-commercial use only
- **Best for**: Research, personal projects, evaluation, non-commercial applications

**Important**: You must accept the model license on Hugging Face before use. See the respective [model cards](https://huggingface.co/black-forest-labs) for full license terms and acceptable use policies.

---

## 🙏 Credits

- **FLUX.2 [klein] Models**: [Black Forest Labs](https://blackforestlabs.ai/)
- **Diffusers Library**: [Hugging Face](https://huggingface.co/docs/diffusers)
- **Gradio Framework**: [Gradio Team](https://gradio.app/)
- **Pinokio**: [Pinokio.computer](https://pinokio.computer/)

---

## 🔗 Links

### Model Cards

**4B Models** (Apache 2.0 License)
- [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) - Full precision
- [FLUX.2 klein 4B NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4) - NVFP4 quantized
- [FLUX.2 klein 4B FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) - FP8 quantized

**9B Models** (Non-Commercial License)
- [FLUX.2 klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) - Full precision
- [FLUX.2 klein 9B NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4) - NVFP4 quantized
- [FLUX.2 klein 9B FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) - FP8 quantized

### Resources
- [Black Forest Labs Official Site](https://blackforestlabs.ai/)
- [Black Forest Labs Blog](https://blackforestlabs.ai/blog)
- [FLUX GitHub Repository](https://github.com/black-forest-labs/flux)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [BFL API](https://bfl.ai/)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub!

---

<div align="center">

Made with ❤️ by the community

**[Report Bug](../../issues)** • **[Request Feature](../../issues)**

</div>
