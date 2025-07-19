import click
import uvicorn

from pipeline.api.health_server import app
from pipeline.telemetry import init_tracing


@click.command()
@click.option('--host', default='127.0.0.1', help='Host interface to bind (default: localhost for security).')
@click.option('--port', default=8000, type=int, help='Port to listen on.')
@click.option('--production', is_flag=True, help='Enable production mode with 0.0.0.0 binding.')
def main(host: str, port: int, production: bool) -> None:
    """Serve the FastAPI app with uvicorn."""
    # Override host for production mode
    if production:
        host = '0.0.0.0'
        click.echo("⚠️  Production mode: binding to all interfaces (0.0.0.0)")
    else:
        click.echo(f"🔒 Development mode: binding to {host}")
    
    init_tracing(app)
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
