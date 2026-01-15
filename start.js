module.exports = {
  daemon: true,
  run: [
    // Launch FLUX.2 klein Gradio Web UI
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { },
        message: [
          "python app.py"
        ],
        on: [{
          // Monitor for Gradio's HTTP URL output
          "event": "/http:\\/\\/[^\\s\\/]+:\\d{2,5}(?=[^\\w]|$)/",
          "done": true
        }]
      }
    },
    // Set the local URL variable for the "Open Web UI" button
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    },
    {
      method: "notify",
      params: {
        html: "FLUX.2 [klein] is running! Click 'Open Web UI' to start generating images. Remember to set your Hugging Face token in the web interface."
      }
    }
  ]
}

