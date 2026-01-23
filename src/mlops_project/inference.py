"""Simple CLI to send images to the API and get predictions."""

import csv
import os
import subprocess
from pathlib import Path

import requests
import typer
from PIL import Image

app = typer.Typer(help="Send images to API and get predictions")


def is_image_file(path: Path) -> bool:
    """Check if file is a valid image."""
    try:
        Image.open(path)
        return path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    except Exception:
        return False


def get_auth_token() -> str | None:
    """Get GCP authentication token for Vertex AI."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@app.command()
def predict(
    folder: str = typer.Argument(..., help="Folder containing images"),
    api_url: str = typer.Option(
        os.getenv("API_URL", "http://localhost:8000"),
        help="API URL (local URL or Vertex AI endpoint ID)",
        envvar="API_URL",
    ),
    output: str = typer.Option("predictions.csv", help="Output CSV file"),
    vertex_ai: bool = typer.Option(False, "--vertex-ai", "-v", help="Use Vertex AI Endpoints format"),
    project: str = typer.Option("mlops-485010", help="GCP project ID (for Vertex AI)"),
    region: str = typer.Option("europe-west1", help="GCP region (for Vertex AI)"),
) -> None:
    """Send images from folder to API and save predictions to CSV."""
    folder_path = Path(folder)
    if not folder_path.exists():
        typer.echo(f"❌ Folder not found: {folder}", err=True)
        raise typer.Exit(1)

    typer.echo(f"📁 Scanning folder: {folder_path}")
    image_files = [f for f in folder_path.iterdir() if f.is_file() and is_image_file(f)]

    if not image_files:
        typer.echo("❌ No image files found", err=True)
        raise typer.Exit(1)

    typer.echo(f"📸 Found {len(image_files)} images")

    # Build the correct URL and headers
    headers = {}
    if vertex_ai or "aiplatform.googleapis.com" in api_url:
        # Vertex AI Endpoints - need auth and different URL format
        token = get_auth_token()
        if not token:
            typer.echo("❌ Failed to get GCP auth token. Run: gcloud auth login", err=True)
            raise typer.Exit(1)
        headers["Authorization"] = f"Bearer {token}"

        # If api_url is just the endpoint ID, build the full URL
        if not api_url.startswith("http"):
            api_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{api_url}:rawPredict"
        typer.echo(f"🌐 Vertex AI URL: {api_url}")
    else:
        typer.echo(f"🌐 API URL: {api_url}")

    results = []
    for i, image_path in enumerate(image_files, 1):
        typer.echo(f"Processing {i}/{len(image_files)}: {image_path.name}...", nl=False)

        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    f"{api_url}/predict" if not vertex_ai and "rawPredict" not in api_url else api_url,
                    files={"file": (image_path.name, f, "image/jpeg")},
                    headers=headers,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

            results.append(
                {
                    "image": image_path.name,
                    "prediction": result["prediction"],
                    "label": result["label"],
                    "confidence": f"{result['confidence']:.4f}",
                    "model_version": result["model_version"],
                }
            )
            typer.echo(f" ✅ {result['label']} ({result['confidence']:.2%})")

        except requests.exceptions.RequestException as e:
            typer.echo(f" ❌ Error: {e}", err=True)
            results.append(
                {
                    "image": image_path.name,
                    "prediction": "error",
                    "label": "error",
                    "confidence": "",
                    "model_version": "",
                }
            )

    typer.echo(f"\n💾 Saving results to {output}")
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "prediction", "label", "confidence", "model_version"])
        writer.writeheader()
        writer.writerows(results)

    typer.echo(f"✅ Done! Results saved to {output}")


if __name__ == "__main__":
    app()
