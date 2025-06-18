import sys
import json
import subprocess


def main(config):
    if config['classifier'] == 'knn':
        subprocess.run(["python", "gnnged/evaluation/knn_classifier.py"])
    elif config['classifier'] == 'svm':
        subprocess.run(["python", "gnnged/evaluation/svm_classifier.py"])
    else:
        raise ValueError(f"Unsupported classifier type: {config['classifier']}")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'params.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)