from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings

# Determine the URL
db_url = settings.DATABASE_URL
if not db_url:
    # Fallback to local default if not set
    db_url = "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/antiplagiarism"

engine = create_async_engine(db_url, echo=False)

async_session = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

async def get_db():
    async with async_session() as session:
        yield session