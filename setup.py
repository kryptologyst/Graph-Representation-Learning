#!/usr/bin/env python3
"""Setup script for graph representation learning project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Graph Representation Learning Project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create necessary directories
    directories = [
        "data",
        "checkpoints", 
        "assets/plots",
        "assets/embeddings",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install pytest black ruff pre-commit", "Installing development dependencies"):
        print("⚠️  Development dependencies installation failed, but continuing...")
    
    # Setup pre-commit hooks
    if Path(".pre-commit-config.yaml").exists():
        if not run_command("pre-commit install", "Setting up pre-commit hooks"):
            print("⚠️  Pre-commit setup failed, but continuing...")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but setup completed")
    
    # Run example
    if not run_command("python scripts/example.py", "Running example script"):
        print("⚠️  Example script failed, but setup completed")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Run the interactive demo: streamlit run demo/streamlit_demo.py")
    print("2. Train a model: python scripts/train.py --model deepwalk --dataset karate")
    print("3. Explore the Jupyter notebook: jupyter notebook notebooks/tutorial.ipynb")
    print("4. Read the documentation: README.md")
    print("\n🔧 Development commands:")
    print("- Format code: black src/ scripts/ demo/")
    print("- Lint code: ruff check src/ scripts/ demo/")
    print("- Run tests: pytest tests/")
    print("- Run all checks: pre-commit run --all-files")


if __name__ == "__main__":
    main()
