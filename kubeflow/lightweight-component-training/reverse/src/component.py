import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Training resusable components')
parser.add_argument('--input-text', type=str, help='The text to reverse.')
args = parser.parse_args()

print(args.input_text)

reversed = args.input_text[::-1]

Path('/text.txt').write_text(reversed)

