#!/usr/bin/env python3
"""
Simple script to run the Streamlit RAG Chat Application
"""

import subprocess
import sys
import os

def main():
    # Change to streamlit_app directory
    streamlit_dir = os.path.join(os.path.dirname(__file__), 'streamlit_app')
    
    if not os.path.exists(streamlit_dir):
        print("❌ Streamlit app directory not found!")
        sys.exit(1)
    
    # Change to streamlit directory
    os.chdir(streamlit_dir)
    
    print("🚀 Starting RAG Chat Application...")
    print("📁 Working directory:", os.getcwd())
    print("🌐 The app will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()