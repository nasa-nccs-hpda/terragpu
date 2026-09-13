"""Independent WorldView validation command."""
import argparse
import json
from terragpu.worldview_validation import validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('item')
    parser.add_argument('output')
    print(json.dumps(validate(**vars(parser.parse_args())), indent=2))
