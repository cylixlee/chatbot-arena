from src.settings import load_environment_settings

CONFIG = load_environment_settings("environment-settings.toml")


def main() -> None:
    print(CONFIG)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
