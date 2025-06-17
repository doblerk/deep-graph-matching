import subprocess
from gnnged.training.train_model import main


def main():
    subprocess.run(["python", "gnnged/training/train_model.py"])


if __name__ == "__main__":
    main()