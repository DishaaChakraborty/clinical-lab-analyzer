import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Get DATABASE_URL from .env, fallback to MySQL connection string
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    # Fallback to MySQL if DATABASE_URL not set
    DATABASE_URL = (
        f"mysql+mysql-connector-python://{os.getenv('DATABASE_USER', 'root')}:"
        f"{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST', 'localhost')}:"
        f"{os.getenv('DATABASE_PORT', '3306')}/{os.getenv('DATABASE_NAME', 'clinical_lab')}"
    )

print(f"DEBUG: Using DATABASE_URL = {DATABASE_URL}")

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv('DEBUG', 'True') == 'True',
    pool_recycle=3600,
)

# Session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base for models
Base = declarative_base()

def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
