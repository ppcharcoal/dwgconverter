"""PyInstaller entry point for the desktop app."""
import multiprocessing

from dwgconverter.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
