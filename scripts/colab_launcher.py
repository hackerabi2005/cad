"""
Colab remote launcher script.
Extracts input bundle, executes training, packages ml/artifacts/ into artifacts.tar.gz.
"""

import os
import subprocess
import tarfile
import sys


def main():
    print("--- [Colab Launcher] Unpacking bundle_input.tar.gz ---")
    if os.path.exists("bundle_input.tar.gz"):
        with tarfile.open("bundle_input.tar.gz", "r:gz") as tar:
            tar.extractall()
        print("Bundle extracted successfully.")
    else:
        print("No bundle_input.tar.gz found, assuming files are in current working directory.")

    print("--- [Colab Launcher] Running ml.train ---")
    ret = subprocess.run([sys.executable, "-m", "ml.train"], check=True)

    print("--- [Colab Launcher] Packaging artifacts ---")
    artifacts_dir = os.path.join("ml", "artifacts")
    with tarfile.open("artifacts.tar.gz", "w:gz") as tar:
        tar.add(artifacts_dir, arcname="artifacts")
    print("Artifacts packaged into artifacts.tar.gz.")


if __name__ == "__main__":
    main()
