module.exports = {
  run: [
    // Install FLUX.2 klein dependencies first (before torch to prevent version conflicts)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "uv pip install -r requirements.txt"
      }
    },
    // Install PyTorch with CUDA support for GPU acceleration
    // FLUX.2 models require GPU with sufficient VRAM (4B: ~13GB, 9B: ~29GB)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: false,
          flashattn: false,
          triton: true
        }
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation complete! Click 'Start' to launch FLUX.2 [klein]. You'll need to set your Hugging Face token in the web interface to download and use the models."
      }
    }
  ]
}
