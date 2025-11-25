"""
Audio-to-Text Transcription Service

This FastAPI application provides speech-to-text transcription using OpenAI's Whisper model.
It includes API key authentication and multi-language support via faster-whisper.
"""

import logging
import torch
import os
from contextlib import asynccontextmanager
from io import BytesIO
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
import uvicorn
from dotenv import load_dotenv
from google.cloud import storage
from pydantic import BaseModel
import numpy as np
import ffmpeg
import whisper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Environment Variables
load_dotenv()
DEVICE = os.environ.get("DEVICE")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "LOCAL")

# Initialize global variables
model = None
storage_client = None
bucket = None

# From: https://github.com/openai/whisper/discussions/908#discussioncomment-5429636
# Aux load bytes function
def load_audio(file_bytes: bytes, sr: int = 16_000) -> np.ndarray:
    """
    Use file's bytes and transform to mono waveform, resampling as necessary
    Parameters
    ----------
    file: bytes
        The bytes of the audio file
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input('pipe:', threads=0)
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .global_args('-loglevel', 'error')
            .run_async(pipe_stdin=True, pipe_stdout=True)
        ).communicate(input=file_bytes)

    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Validate API key from request header.

    Ensures that incoming requests include a valid API key in the X-API-Key header
    for secure access to the transcription service.

    Args:
        api_key: The API key provided in the request header

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key required. Please provide X-API-Key header."
        )

    if api_key != os.environ.get("AUDIO_TO_TEXT_API_KEY", ""):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    return api_key


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Manage application lifespan events.

    On startup, initializes the Whisper model with appropriate device and compute type
    based on the MODEL_SIZE environment variable. Uses CUDA and float16 for large models,
    CPU and int8 for smaller models to optimize performance and memory usage.

    Also initializes the GCP storage client for accessing audio files from Cloud Storage.
    """
    global model, storage_client, bucket
    
    if ENVIRONMENT == 'PROD':
        model = whisper.load_model("models/large-v3-turbo.pt")
    else:
        model = whisper.load_model("models/small.pt")
    
    # Initialize GCP storage client
    try:
        storage_client = storage.Client(project=PROJECT_NAME)
        bucket = storage_client.bucket(BUCKET_NAME)
    except Exception as e:
        logger.warning(f"Failed to initialize GCP storage client: {e}")

    yield


# Response model for transcription results
class TranscriptionResponse(BaseModel):
    """Response model for successful transcription."""
    detected_language: str
    language_probability: float
    transcription: str


app = FastAPI(lifespan=lifespan)
app.title = "Reflection Journal - Audio To Text"
app.version = "0.0.2"


@app.get(
    "/",
    tags=["Root"],
    summary="Welcome Endpoint",
    description="Returns a welcome message including the application title and version."
)
def root():
    """Root endpoint to verify the service is running."""
    return {"message": f"Welcome to {app.title} v{app.version}"}


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Checks if the model and GCP bucket are properly initialized."
)
def health():
    """
    Health check endpoint that verifies critical resources are available.

    Returns:
        dict: Contains a 'healthy' boolean and status of each service (model, bucket).
    """
    model_ready = model is not None
    bucket_ready = bucket is not None

    return {
        "healthy": model_ready and bucket_ready,
        "services": {
            "model": "available" if model_ready else "unavailable",
            "bucket": "available" if bucket_ready else "unavailable"
        },
        "cuda": torch.cuda.is_available()
    }


@app.post(
    "/transcribe",
    tags=["Transcription"],
    summary="Transcribe Audio to Text",
    description="Converts speech in audio files to text using faster-whisper. "
                "Reads audio files from GCP Cloud Storage and auto-detects language.",
    response_model=TranscriptionResponse,
    dependencies=[Security(verify_api_key)]
)
async def transcribe(audio_id: str):
    """
    Transcribe audio file from GCP Cloud Storage to text.

    Fetches an audio file from Cloud Storage (path: audio/{audio_id}) and converts it
    to text using the faster-whisper model. Automatically detects the language of the
    audio and returns the transcription along with language information.

    Args:
        audio_id: The identifier of the audio file in Cloud Storage (path: audio/{audio_id})

    Returns:
        TranscriptionResponse: Contains the transcribed text, detected language, and confidence

    Raises:
        HTTPException: 400 if audio_id is empty or model not initialized
        HTTPException: 404 if audio file not found in bucket
        HTTPException: 500 if transcription fails
        HTTPException: 401 if API key is invalid
    """
    if not model:
        raise HTTPException(
            status_code=500,
            detail="Whisper model not initialized. Please try again later."
        )

    if not audio_id or not audio_id.strip():
        raise HTTPException(
            status_code=400,
            detail="audio_id parameter is required and cannot be empty."
        )

    if not bucket:
        raise HTTPException(
            status_code=500,
            detail="GCP storage bucket not initialized. Please try again later."
        )

    try:
        # Construct the blob path
        blob_path = f"audio/{audio_id}"
        blob = bucket.blob(blob_path)

        # Check if blob exists
        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found: {blob_path}"
            )

        # Download file content to memory
        audio_bytes = blob.download_as_bytes()

        # Load and process audio
        audio = load_audio(audio_bytes)
        audio = whisper.pad_or_trim(audio)

        # Create log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=128 if ENVIRONMENT == "PROD" else 80
        ).to(model.device)

        # Detect the spoken language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)  # type: ignore
        language_probability = probs[detected_language]  # type: ignore

        # Decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        return {
            "detected_language": detected_language,
            "language_probability": language_probability,
            "transcription": result.text  # type: ignore
        }
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe audio: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
