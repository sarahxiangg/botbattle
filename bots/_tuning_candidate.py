import os

config_path = os.environ.get("CANDIDATE_CONFIG")
if not config_path:
    raise RuntimeError("CANDIDATE_CONFIG was not provided")

os.environ["BOT_CONFIG"] = config_path

from my_bot import main

if __name__ == "__main__":
    main()
