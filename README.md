# AI-Call-Center-Prototype-Faster-Whisper-TinyLLaMA-Demo
# Overview
This prototype demonstrates a two-stage AI pipeline designed for call center applications. It leverages:

Faster-Whisper for fast and accurate audio transcription.

TinyLLaMA for generating text responses based on the transcribed content.

This solution is optimized for low-latency inference, making it suitable for real-time customer support scenarios.

# Features
Fast Transcription: Uses Faster-Whisper with FP16 computation for speedy audio processing.

Real-time Text Generation: Employs TinyLLaMA for generating concise responses.

GPU Acceleration: Automatically detects and uses GPU (if available) to further improve performance.

Modular Design: The code is structured for easy integration and further customization.

# Requirements
Python 3.8+

PyTorch

Faster-Whisper

Transformers (by Hugging Face)

You can install the dependencies using pip:


pip install torch faster-whisper transformers
Usage
Prepare Your Audio File:
Ensure your audio file (e.g., one containing the phrase "check my balance") is available. Update the audio_path variable in the script to point to your file location.

Run the Script:
Execute the script via the command line:


python your_script_name.py

# Review the Output:
The program prints:

The device being used (GPU/CPU).

Transcription time and the transcribed text.

Text generation time and the generated response.

Total inference time.

Code Structure
Transcription Section:
Loads the "tiny.en" variant of Faster-Whisper, performs transcription on the provided audio file, and prints the transcribed text along with the inference time.

Text Generation Section:
Loads the TinyLLaMA model and its tokenizer. It then uses the transcribed text as input to generate a response, displaying the generation time and final output.

Customization Options
Audio File Path:
Change the audio_path variable to the location of your desired audio input.

Model Variants:
To experiment with different performance profiles, consider swapping out model names in both Faster-Whisper and TinyLLaMA.

Generation Parameters:
Modify parameters such as max_new_tokens to adjust the length of the generated text.

Future Enhancements
Implement error handling and logging for improved robustness.

Modularize the code for integration into larger call center systems.

Explore options for real-time audio streaming and processing.



# Contact
For questions or further discussion, please contact sriram rampelli at sriramrampelli15@gmail.com .

