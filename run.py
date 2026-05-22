"""Simple entrypoint for codex_stripper proxy."""

from dotenv import load_dotenv

load_dotenv()

from codex_stripper.proxy import main

if __name__ == "__main__":
    main()
