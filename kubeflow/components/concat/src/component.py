import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Concatenate text')
parser.add_argument('--first',  type=str, help='The first text.')
parser.add_argument('--second', type=str, help='The second text.')
args = parser.parse_args()

print(args.first)
print(args.second)

concat = args.first + args.second

Path('/text.txt').write_text(concat)

