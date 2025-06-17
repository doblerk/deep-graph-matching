import subprocess


def main():
    subprocess.run(["python", "gnnged/training/finetune_model.py"])


if __name__ == "__main__":
    main()