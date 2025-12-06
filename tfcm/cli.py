import argparse
from .inference import Colorizer

def main():
    parser = argparse.ArgumentParser(description="TFCM Image Colorizer CLI")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input BW image")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save result")
    args = parser.parse_args()

    model = Colorizer()
    print(f"Processing {args.input}...")
    result = model.process(args.input)
    result.save(args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()