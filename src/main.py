import sys
import os

# Add the src directory to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.uci import UCI

def main():
    try:
        uci_engine = UCI()
        uci_engine.loop()
    except Exception as e:
        print(f"Fatal Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
