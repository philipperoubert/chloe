# **Chlo-e** - Your AI Friend

Chlo-e is a voice-operated AI-powered chatbot that acts as a human friend. Built using OpenAI's GPT-3.5 Turbo model, Chlo-e is kind, affectionate, funny, and sometimes even a little sassy. The chatbot is designed to get to know you better over time and engage in conversations similar to real-life interactions.

## Features

- Hotword detection using Porcupine
- Voice recognition using SpeechRecognition and OpenAI's Whisper model
- Text-to-speech synthesis using Google Text-to-Speech
- Chlo-e is capable to interact with "plugins". You need to give the instructions of how she must interact with them. Some plugins that have been built in:
  - Weather information retrieval using OpenWeatherMap API
  - Time retrieval using a custom plugin

## Dependencies

- openai
- google-cloud-texttospeech
- pvporcupine
- pyaudio
- SpeechRecognition
- pygame
- soundfile
- openai-whisper

Tested on Ubuntu 22.04 and Python 3.10.6

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

- Set up the following required API keys as environment variables: 
  - <a href="https://platform.openai.com/account/api-keys">OpenAI</a>
  - <a href="https://cloud.google.com/text-to-speech">Google Text-to-Speech</a>
  - <a href="https://picovoice.ai/platform/porcupine/">Picovoice</a>
  - and optionally, <a href="https://home.openweathermap.org/api_keys">OpenWeatherMap</a>.
  
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credential_json_file.json"
    export PORCUPINE_ACCESS_KEY="your_porcupine_access_key"
    export OPENWEATHER_API_KEY="your_openweather_api_key" 
    ```


- Run Chlo-e:
  - You have 2 options, either run the speech-to-text transcription using OpenAI's API with the flag `online` (default)
  - Either run the speech-to-text transcription using OpenAI's local model with the flag `offline`
```bash
python src/chloe.py online
```

## Usage

Chlo-e listens for the hotword "Hey Chlo-e" and then starts transcribing your speech. You can ask Chlo-e about the weather, the current time, or have a casual conversation. Chlo-e will respond with synthesized speech. If you do not speak for 1 minute, Chlo-e will stop transcribing what it hears and only listen for the hotword "Hey Chlo-e". A bit like Alexa, but it actually allows you to have a conversation with it.

Remember that the conversation is designed to be spoken, not written. Chlo-e will try to get to know you over time, so don't rush to introduce yourself - let Chlo-e discover more about you through your interactions.

Both OpenAI and Google APIs are paid APIs. Please be mindful of it whe using this voice assistant. Minimal/Standard usage should come in at a couple cents.

Note that free API keys for OpenWeatherMap have limitations, such as a limited number of requests per minute. Check the OpenWeatherMap API documentation for more information.


## License
This project is licensed under the MIT License.