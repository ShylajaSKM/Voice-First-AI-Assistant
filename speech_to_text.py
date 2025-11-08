import os
import wave
import json
import requests
import sounddevice as sd
from dotenv import load_dotenv
import chainlit as cl
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Helper to read Deepgram API key
def get_deepgram_key():
    return os.getenv("DEEPGRAM_API_KEY")

recording_in_progress = False

def record_audio(duration=5, sample_rate=16000):
    """Record audio from the microphone."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    return audio_data, sample_rate

def save_audio_to_wav(audio_data, sample_rate, filename="temp_audio.wav"):
    """Save audio data to a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return filename

@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content=(
            "# Welcome to the Shy's Text to Speech AI Assistant!\n\n"
            "**Click the record button to start speaking.**"
        ),
        actions=[
            cl.Action(
                name="Start Recording",
                payload={"cmd": "record", "duration": 5},
                description="Begin a 5-second recording",
            ),
            cl.Action(
                name="Stop Recording",
                payload={"cmd": "stop"},
                description="Stop the current recording",
            ),
        ],
    ).send()

async def send_action_bar():
    await cl.Message(
        content="",
        actions=[
            cl.Action(
                name="Start Recording",
                payload={"cmd": "record", "duration": 5},
                description="Begin a 5-second recording",
            ),
            cl.Action(
                name="Stop Recording",
                payload={"cmd": "stop"},
                description="Stop the current recording",
            ),
        ],
    ).send()

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "[Missing GOOGLE_API_KEY. Set it in .env]"
    if not getattr(call_gemini, "_configured", False):
        genai.configure(api_key=api_key)
        call_gemini._configured = True
    # Primary model from env or sensible default
    primary = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    fallbacks = [
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro-latest",
    ]
    models_to_try = [primary] + [m for m in fallbacks if m != primary]
    last_err = None
    for name in models_to_try:
        try:
            model = genai.GenerativeModel(name)
            resp = model.generate_content(prompt)
            if hasattr(resp, "text") and resp.text is not None:
                return resp.text
        except Exception as e:
            last_err = e
            continue
    return f"[Gemini error: {last_err}]"

def tts_deepgram(text: str, voice: str = None) -> bytes:
    api_key = get_deepgram_key()
    if not api_key:
        return b""
    voice = voice or os.getenv("DEEPGRAM_VOICE", "aura-asteria")
    url = "https://api.deepgram.com/v1/speak"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }
    params = {"model": voice}
    body = {"text": text}
    r = requests.post(url, headers=headers, params=params, data=json.dumps(body), timeout=60)
    if r.status_code != 200:
        return b""
    return r.content

@cl.action_callback("Start Recording")
async def on_record(action: cl.Action):
    # Pull options from payload when available
    payload = getattr(action, "payload", {}) or {}
    duration = int(payload.get("duration", 5))
    sample_rate = int(payload.get("sample_rate", 16000))
    global recording_in_progress
    recording_in_progress = True
    await cl.Message("Recording... Please speak now.").send()
    audio_data, sample_rate = record_audio(duration, sample_rate)
    recording_in_progress = False
    audio_file = save_audio_to_wav(audio_data, sample_rate)
    api_key = get_deepgram_key()
    if not api_key:
        await cl.Message(
            content=(
                "DEEPGRAM_API_KEY is not set. Add it to a .env file or your environment and restart."
            ),
        ).send()
        return
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    params = {
        "model": "nova-2",
        "smart_format": "true",
        "punctuate": "true",
    }
    resp = requests.post(
        "https://api.deepgram.com/v1/listen",
        headers=headers,
        params=params,
        data=audio_bytes,
        timeout=60,
    )
    if resp.status_code != 200:
        await cl.Message(content=f"Deepgram error: {resp.status_code} {resp.text}").send()
        return
    data = resp.json()
    transcript = (
        data.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
    )
    reply = call_gemini(f"User said: {transcript}. Provide a helpful, concise spoken reply.")
    audio_reply = tts_deepgram(reply)
    await cl.Message(
        content=f"You: {transcript}\n\nAssistant: {reply}",
        elements=[
            cl.Audio(name="reply.wav", content=audio_reply, mime="audio/wav") if audio_reply else None
        ],
    ).send()
    await send_action_bar()

@cl.on_message
async def process_audio(message: cl.Message):
    if not message.elements and (message.content or "").strip():
        transcript = (message.content or "").strip()
        reply = call_gemini(f"User said: {transcript}. Provide a helpful, concise spoken reply.")
        audio_reply = tts_deepgram(reply)
        await cl.Message(
            content=f"You: {transcript}\n\nAssistant: {reply}",
            elements=[
                cl.Audio(name="reply.wav", content=audio_reply, mime="audio/wav") if audio_reply else None
            ],
        ).send()
        await send_action_bar()
        return
    if not message.elements:
        # Record audio if no file is uploaded
        duration = 5  # seconds
        sample_rate = 16000
        
        await cl.Message("Recording... Please speak now.").send()
        audio_data, sample_rate = record_audio(duration, sample_rate)
        
        # Save the recorded audio
        audio_file = save_audio_to_wav(audio_data, sample_rate)
        
        # Transcribe the audio via Deepgram
        api_key = get_deepgram_key()
        if not api_key:
            await cl.Message(
                content=(
                    "DEEPGRAM_API_KEY is not set. Add it to a .env file or your environment and restart."
                ),
            ).send()
            return
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        }
        params = {
            "model": "nova-2",
            "smart_format": "true",
            "punctuate": "true",
        }
        resp = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers=headers,
            params=params,
            data=audio_bytes,
            timeout=60,
        )
        if resp.status_code != 200:
            await cl.Message(content=f"Deepgram error: {resp.status_code} {resp.text}").send()
            return
        data = resp.json()
        transcript = (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
        reply = call_gemini(f"User said: {transcript}. Provide a helpful, concise spoken reply.")
        audio_reply = tts_deepgram(reply)
        await cl.Message(
            content=f"You: {transcript}\n\nAssistant: {reply}",
            elements=[
                cl.Audio(name="reply.wav", content=audio_reply, mime="audio/wav") if audio_reply else None
            ],
        ).send()
    else:
        # Handle uploaded audio files
        for element in message.elements:
            if element.mime.startswith('audio/'):
                # Save the uploaded file
                audio_file = "uploaded_audio.wav"
                with open(audio_file, "wb") as f:
                    f.write(element.content)
                
                # Transcribe the audio via Deepgram
                api_key = get_deepgram_key()
                if not api_key:
                    await cl.Message(
                        content=(
                            "DEEPGRAM_API_KEY is not set. Add it to a .env file or your environment and restart."
                        ),
                    ).send()
                    return
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                headers = {
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "audio/wav",
                }
                params = {
                    "model": "nova-2",
                    "smart_format": "true",
                    "punctuate": "true",
                }
                resp = requests.post(
                    "https://api.deepgram.com/v1/listen",
                    headers=headers,
                    params=params,
                    data=audio_bytes,
                    timeout=60,
                )
                if resp.status_code != 200:
                    await cl.Message(content=f"Deepgram error: {resp.status_code} {resp.text}").send()
                    return
                data = resp.json()
                transcript = (
                    data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                )
                
                # Send the transcription back to the user
                await cl.Message(
                    content=f"Transcription: {transcript}",
                ).send()
                break

if __name__ == "__main__":
    print("This is a Chainlit app. Start it with: chainlit run speech_to_text.py -w")
    print("Ensure you set DEEPGRAM_API_KEY in a .env file or environment.")
