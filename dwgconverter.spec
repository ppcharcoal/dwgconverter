# PyInstaller spec: one folder with the GUI (DWGConverter.exe) and the
# command line tool (dwgconvert.exe) sharing the same libraries.
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hidden = collect_submodules("ezdxf") + ["skimage.morphology", "skimage.measure"]
excludes = ["matplotlib", "pandas", "IPython", "pytest", "PyQt5", "PySide6", "notebook"]

a_gui = Analysis(["app_gui.py"], hiddenimports=hidden, excludes=excludes)
a_cli = Analysis(["app_cli.py"], hiddenimports=hidden, excludes=excludes)

pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)

exe_gui = EXE(pyz_gui, a_gui.scripts, [], exclude_binaries=True, name="DWGConverter",
              console=False, icon=None)
exe_cli = EXE(pyz_cli, a_cli.scripts, [], exclude_binaries=True, name="dwgconvert",
              console=True)

coll = COLLECT(exe_gui, a_gui.binaries, a_gui.datas,
               exe_cli, a_cli.binaries, a_cli.datas,
               name="DWGConverter")
