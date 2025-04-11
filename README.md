silly little python pipeline for live translation. (whisper-stream didn't work for korean)

microphone input -> whisper.cpp -> llama.cpp -> translated output

### Requirements

* running whisper.cpp server (example: `$WHISPER/build/bin/whisper-server -m $WHISPER/models/ggml-large-v3-turbo.bin --host 127.0.0.1 --port 9000 -t 8 --language ko -pp -pr -sow -mc 768`)
* running llama.cpp server
