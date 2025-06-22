import click
import uvicorn

from pipeline.api.health_server import app
from pipeline.telemetry import init_tracing


@click.command()
@click.option('--host', default='0.0.0.0', help='Host interface to bind.')
@click.option('--port', default=8000, type=int, help='Port to listen on.')
def main(host: str, port: int) -> None:
    """Serve the FastAPI app with uvicorn."""
    init_tracing(app)
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
