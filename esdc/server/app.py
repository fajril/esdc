# Standard library
import logging

# Third-party
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Local
from esdc.configs import Config
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
        """Handle generic exceptions."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error"}},
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

    # Create FastAPI app
    app = create_app()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    logger.info(f"Starting ESDC server on http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )
