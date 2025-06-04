import subprocess


def main():
    subprocess.run(["python", "gnn_ged/training/finetune_model.py"])
    subprocess.run(["python", "gnn_ged/training/train_model.py"])


if __name__ == "__main__":
    main()