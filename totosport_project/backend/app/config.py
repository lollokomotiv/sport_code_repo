from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    database_url: str

    # Auth
    secret_key: str
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # API-Football (opzionale per ora)
    api_football_key: str = ""
    api_football_host: str = "v3.football.api-sports.io"

    # CORS
    allowed_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignora variabili extra nel .env (es. POSTGRES_USER per Docker)
    )


settings = Settings()
