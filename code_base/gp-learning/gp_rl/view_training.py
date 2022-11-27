from loguru import logger
from utils.data_loading import load_data


def main():
    data = load_data("./results/_trial_1.pkl")
    logger.info(data)


if __name__ == "__main__":
    main()
