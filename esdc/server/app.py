# Standard library
import logging

import uvicorn

# Third-party
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Local
from esdc.configs import Config
from esdc.phoenix import setup_phoenix_tracing
from esdc.server.logging_config import setup_server_logging
from esdc.server.routes import router

# Logger
logger = logging.getLogger("esdc.server")


def create_app() -> FastAPI:
    """Create FastAPI application with CORS middleware.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="ESDC API",
        description="OpenAI-compatible API for ESDC agent",
        version="0.4.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes with /v1 prefix
    app.include_router(router, prefix="/v1")

    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        """Handle generic exceptions with full context."""
        import traceback

        error_msg = f"Unhandled exception: {type(exc).__name__}: {str(exc)}"
        stack_trace = traceback.format_exc()

        logger.error(f"[EXCEPTION] {error_msg}")
        logger.error(f"[STACK TRACE]\n{stack_trace}")

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "detail": "Check server logs for full traceback",
                }
            },
        )

    return app


def run_server(
    host: str = "0.0.0.0", port: int = 3334, log_level: str = "info"
) -> None:
    """Run the web server.

    Args:
        host: Server host address
        port: Server port
        log_level: Uvicorn log level
    """
    # Initialize configuration
    Config.init_config()
    setup_phoenix_tracing()

    # Setup proper logging configuration
    setup_server_logging(
        {
            "level": "DEBUG",
            "server": {
                "level": "DEBUG",
                "file": {
                    "enabled": True,
                    "path": "logs/esdc_server.log",
                    "max_size": "10MB",
                    "backup_count": 5,
                },
                "console": {
                    "enabled": True,
                    "level": "DEBUG",
                },
            },
        }
    )

    # Create FastAPI app
    app = create_app()

    logger.info(f"Starting ESDC server on http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )
