import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import Session, select
from models import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv()

# Settings
class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///db/reflexion.db")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
    AI_WORKER_URL: str = os.getenv("AI_WORKER_URL", "http://ai-worker:8080")
    AI_WORKER_API_KEY: str = os.getenv("AI_WORKER_API_KEY", "your-secret-key-change-this-in-production")
    GOOGLE_CLOUD_PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")
    PUBSUB_SUBSCRIPTION: str = os.getenv("PUBSUB_SUBSCRIPTION", "")
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "465"))
    SENDER_EMAIL: str = os.getenv("SENDER_EMAIL", "")
    SENDER_PASSWORD: str = os.getenv("SENDER_PASSWORD", "").strip('"')  # Remove quotes if present
    RECIPIENT_EMAIL: str = os.getenv("RECIPIENT_EMAIL", "")

# Global settings
settings = Settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.JWT_SECRET_KEY if hasattr(settings, 'JWT_SECRET_KEY') else "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# HTTP Bearer token dependency (returns 401 instead of 403)
security = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""
    # Bcrypt has a maximum password length of 72 bytes
    truncated_password = plain_password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    return pwd_context.verify(truncated_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    # Bcrypt has a maximum password length of 72 bytes
    truncated_password = password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    return pwd_context.hash(truncated_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_user(email: str, password: str, session: Session) -> Optional[User]:
    """Authenticate user with email and password."""
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def verify_token(token: str, session: Session) -> Optional[User]:
    """Verify JWT token and return user."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = str(payload.get("sub"))
        if email is None:
            return None
    except JWTError:
        return None
    
    user = session.exec(select(User).where(User.email == email)).first()
    return user


def get_current_user_dep(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Get current user dependency with database session access."""
    from fastapi_app import database_engine
    
    with Session(database_engine) as session:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        # Check if credentials are provided
        if credentials is None:
            raise credentials_exception
        
        user = verify_token(credentials.credentials, session)
        if user is None:
            raise credentials_exception
        return user
