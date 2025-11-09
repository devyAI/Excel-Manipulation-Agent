import argparse
from agent.pipeline import run_transformation


def parse_args():
    parser = argparse.ArgumentParser(description="Excel Manipulation Agent")
    parser.add_argument("input", help="Path to input Excel file (.xlsx)")
    parser.add_argument("instruction", help="Natural language instruction")
    parser.add_argument("--output", help="Optional output Excel path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, summary = run_transformation(args.input, args.instruction, args.output)
    print(output_path)
    print(summary)


if __name__ == "__main__":
    main()
