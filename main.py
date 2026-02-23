import argparse
from src.preprocessing.preprocess import run
from src.config.config import PreprocessingConfig, TokenizerConfig

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Main controller")
    ap.add_argument("--prepare_data", action="store_true")
    
    return ap.parse_args()

def main():
    args = parse_args()
    if args.prepare_data:
        run(preprocess_config=PreprocessingConfig(), token_config=TokenizerConfig())


if __name__ == '__main__':
    main()