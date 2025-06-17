import json
import sys
from gnnged.utils.split_dataset import main


def run():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'params.json'

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config file '{config_path}': {e}")
        sys.exit(1)

    main(config)

if __name__ == '__main__':
    run()