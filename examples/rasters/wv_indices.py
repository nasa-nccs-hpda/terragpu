"""Windowed WorldView processing; supply the actual source band order."""
import argparse
from terragpu.streaming import process_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--bands', nargs='+', required=True)
    parser.add_argument('--indices', nargs='+', default=['ndvi', 'ndwi'])
    parser.add_argument('--backend', choices=['numpy', 'cupy'], default='numpy')
    parser.add_argument('--tile-size', type=int, default=1024)
    print(process_indices(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
