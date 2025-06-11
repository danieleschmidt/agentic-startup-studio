# core/deployment_manager.py
import time
import re  # For slugifying app name
import os  # For __main__ example
import shutil  # For __main__ example


def deploy_to_fly_io(project_path: str, app_name: str) -> str:
    """
    Simulates deploying a project from project_path to Fly.io with the given app_name.
    In a real scenario, this would involve running 'flyctl deploy' or using Fly.io APIs.

    Args:
        project_path: Path to the project to be deployed.
        app_name: Desired application name on Fly.io.

    Returns:
        A mock application URL if deployment is "successful", otherwise an empty string.
    """
    print(
        f"Simulating deployment of project at '{project_path}' to Fly.io as "
        f"app: '{app_name}'..."
    )

    # Basic validation for inputs
    if not project_path or not app_name:
        print("Error: Project path and app name are required for deployment.")
        return ""

    # Simulate some deployment delay
    time.sleep(1)

    # Mock URL construction (Fly.io typically uses .fly.dev)
    # Slugify app_name to be safe for URLs/hostnames
    safe_app_name = re.sub(r"[^a-z0-9-]+", "-", app_name.lower()).strip("-")
    if not safe_app_name:  # Handle case where app_name results in empty slug
        # Fallback to a generic name with a hash if slug is empty
        safe_app_name = f"app-{abs(hash(app_name)) % 10000}"

    mock_app_url = f"https://{safe_app_name}.fly.dev"
    print(f"Mock deployment successful. Application URL: {mock_app_url}")
    return mock_app_url


def check_fly_io_health(app_name: str) -> bool:
    """
    Simulates checking the health of a deployed Fly.io application.
    In a real scenario, this might involve HTTP checks or 'flyctl status'.

    Args:
        app_name: The name of the Fly.io application.

    Returns:
        True if the application is "healthy", False otherwise.
    """
    print(f"Simulating health check for Fly.io app: '{app_name}'...")
    if not app_name:
        print("Error: App name is required for health check.")
        return False

    # Simulate some delay for health check
    time.sleep(0.5)

    # For this mock, assume it's always healthy if an app_name is provided
    print(f"Mock health check for '{app_name}' successful.")
    return True


if __name__ == "__main__":
    # Basic demonstration
    test_project_path = "./dummy_project_for_deploy"
    if not os.path.exists(test_project_path):
        os.makedirs(test_project_path)

    test_app_name = "my-test-startup-app-123"
    print(f"\n--- Testing with app name: {test_app_name} ---")
    url = deploy_to_fly_io(test_project_path, test_app_name)
    if url:
        # Health check uses original name for this mock
        is_healthy = check_fly_io_health(test_app_name)
        print(f"App: {test_app_name}, URL: {url}, Healthy: {is_healthy}")

    # Test with potentially problematic app name
    tricky_app_name = "My Awesome App!@#"
    print(f"\n--- Testing with tricky app name: {tricky_app_name} ---")
    url2 = deploy_to_fly_io(test_project_path, tricky_app_name)
    if url2:
        # Health check should ideally use the name Fly.io knows the app by,
        # which might be the slugified version or the original if Fly.io handles it.
        # For this mock, let's assume health check uses the original name provided.
        is_healthy2 = check_fly_io_health(tricky_app_name)
        print(f"App: {tricky_app_name}, URL: {url2}, Healthy: {is_healthy2}")

    if os.path.exists(test_project_path):  # Clean up dummy dir
        shutil.rmtree(test_project_path)
    print(f"\nCleaned up temporary directory: {test_project_path}")
    print("--- Deployment Manager Demo Finished ---")
