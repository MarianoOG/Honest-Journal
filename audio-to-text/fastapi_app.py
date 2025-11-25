"""
Audio-to-Text Transcription Service

This FastAPI application provides speech-to-text transcription using OpenAI's Whisper model.
It includes API key authentication and multi-language support via faster-whisper.
"""

import logging
import torch
import os
import secrets
from contextlib import asynccontextmanager
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
AUDIO_TO_TEXT_API_KEY = os.environ.get("AUDIO_TO_TEXT_API_KEY")
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
    for secure access to the transcription service. Uses timing-attack-safe comparison.

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

    expected_key = os.environ.get("AUDIO_TO_TEXT_API_KEY", "")
    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    return api_key


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Manage application lifespan events.

    On startup, validates required environment variables, initializes the Whisper model,
    and sets up the GCP storage client for accessing audio files from Cloud Storage.
    """
    global model, storage_client, bucket

    # Validate required environment variables
    logger.info("Validating required environment variables...")
    required_vars = {
        "AUDIO_TO_TEXT_API_KEY": AUDIO_TO_TEXT_API_KEY,
        "PROJECT_NAME": PROJECT_NAME,
        "BUCKET_NAME": BUCKET_NAME,
    }

    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Configuration: PROJECT_NAME={PROJECT_NAME}, BUCKET_NAME={BUCKET_NAME}")

    # Load Whisper model
    try:
        model_name = "large-v3-turbo" if ENVIRONMENT == 'PROD' else "small"
        model_path = f"models/{model_name}.pt"

        # Check if model file exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}. Attempting to load model...")

        logger.info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_path)
        logger.info(f"Successfully loaded Whisper model: {model_name}")
    except Exception as e:
        error_msg = f"Failed to load Whisper model: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Initialize GCP storage client
    try:
        logger.info("Initializing GCP storage client...")
        storage_client = storage.Client(project=PROJECT_NAME)
        bucket = storage_client.bucket(BUCKET_NAME)
        logger.info(f"Successfully initialized GCP storage client for bucket: {BUCKET_NAME}")
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
def transcribe(audio_id: str):
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
        logger.error("Whisper model not initialized")
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
        logger.error("GCP storage bucket not initialized")
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

        logger.info(f"Transcription successful for audio_id: {audio_id}. Language: {detected_language}")

        return {
            "detected_language": detected_language,
            "language_probability": language_probability,
            "transcription": result.text  # type: ignore
        }
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to transcribe audio ({audio_id}): {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe audio: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
