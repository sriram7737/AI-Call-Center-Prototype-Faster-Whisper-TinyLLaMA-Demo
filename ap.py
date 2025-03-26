import time
import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use GPU if available.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- Transcription using Faster-Whisper ---
# "tiny.en" is the English-specific tiny model variant.
# compute_type "float16" leverages FP16 for speed.
model_whisper = WhisperModel("tiny.en", device=device, compute_type="float16")

audio_path = "d:/traceai/short.wav"  # Your audio file containing "check my balance"
start_time = time.time()
# beam_size=2 helps improve accuracy with minimal overhead.
segments, info = model_whisper.transcribe(audio_path, beam_size=2)
transcription_time = time.time() - start_time

# Concatenate segments to form the full transcription.
transcribed_text = " ".join([seg.text for seg in segments]).strip()
print(f"Faster-Whisper transcription time: {transcription_time:.4f}s")
print("Transcribed Text:", transcribed_text)

# --- Text Generation with TinyLLaMA ---
# Load the TinyLLaMA model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
# Convert to FP16 for faster inference.
model = model.half()

# For very fast generation, you can reduce max_new_tokens.
input_ids = tokenizer.encode(transcribed_text, return_tensors="pt").to(device)
start_time = time.time()
output_ids = model.generate(input_ids, max_new_tokens=1)
generation_time = time.time() - start_time
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"TinyLLaMA generation time: {generation_time:.4f}s")
print("Generated Response:", generated_text)

total_time = transcription_time + generation_time
print(f"Total inference time: {total_time:.4f}s")
