from src.preprocessing.preprocess import run
from src.config.config import PreprocessingConfig, TokenizerConfig


def main():
    run(preprocess_config=PreprocessingConfig(), token_config=TokenizerConfig())


if __name__ == '__main__':
    main()