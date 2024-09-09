# cmo_instance.py
#
# Command Modern Operations instance manager
#
# Author: Giacomo Del Rio
# Creation date: 5 November 2021

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional, Union

import psutil


class CMOInstance:
    CMO_EXECUTABLE = r"C:\Program Files (x86)\Command Professional Edition 2\CommandCLI.exe"
    CMO_DEFAULT_OUTPUT_FOLDER_I = r"C:\ProgramData\Command Professional Edition 2\Analysis_Int"
    CMO_DEFAULT_OUTPUT_FOLDER_MC = r"C:\ProgramData\Command Professional Edition 2\Analysis_MC"
    CMO_DEFAULT_PORT = 7777
    ANTI_CLASH_WAIT = 6

    def __init__(self, scenario: Union[str, Path], port: Optional[int] = None, mode: str = "I",
                 out_folder: Optional[str] = None, executable: Optional[str] = None, autoexit: bool = True,
                 iterations: int = 1, savemessagelog: bool = False, recinterval: Optional[int] = None,
                 startup_time_secs: int = 10, anti_clash_wait: Optional[int] = None):
        self.scenario = scenario
        self.port = port if port is not None else CMOInstance.CMO_DEFAULT_PORT
        self.mode = mode
        self.out_folder = out_folder
        self.executable = executable if executable is not None else CMOInstance.CMO_EXECUTABLE
        self.autoexit = autoexit
        self.iterations = iterations
        self.savemessagelog = savemessagelog
        self.recinterval = recinterval
        self.startup_time_secs = startup_time_secs
        self.anti_clash_wait = anti_clash_wait if anti_clash_wait is not None else CMOInstance.ANTI_CLASH_WAIT

        self.popen_obj: Optional[subprocess.Popen] = None

    def run(self, new_console=True) -> ():
        if self.is_running():
            raise RuntimeError(f"CMO({self.port}) is already running")

        # Avoid startup clashes when multiple instances are spawned at the same time (required as CMO 2.1.7)
        time.sleep(self.anti_clash_wait * (self.port - self.CMO_DEFAULT_PORT))

        self.popen_obj = None
        command = f'"{self.executable}" -port {self.port} -mode {self.mode} ' \
                  f'{f"-autoexit " if self.autoexit else ""}' \
                  f'{f"-iterations {self.iterations} " if self.mode == "MC" else ""}' \
                  f'{f"-savemessagelog " if self.savemessagelog else ""}' \
                  f'{f"-recinterval {self.recinterval} " if self.recinterval else ""}' \
                  f'{f"-outputfolder {self.out_folder} " if self.out_folder else ""}' \
                  f'-scenfile "{self.scenario}"'
        if new_console:
            self.popen_obj = subprocess.Popen(args=command, creationflags=subprocess.CREATE_NEW_CONSOLE,
                                              startupinfo=subprocess.STARTUPINFO(
                                                  dwFlags=subprocess.STARTF_USESHOWWINDOW,
                                                  wShowWindow=6))
        else:
            self.popen_obj = subprocess.Popen(args=command)
        time.sleep(self.startup_time_secs)

    def is_running(self) -> bool:
        if self.popen_obj:
            return self.popen_obj.poll() is None
        else:
            return False

    def wait(self) -> ():
        self.popen_obj.wait()

    def kill(self):
        if self.popen_obj:
            self.popen_obj.kill()
            self.popen_obj = None

    def pid(self) -> Optional[int]:
        if self.popen_obj:
            return self.popen_obj.pid
        else:
            return None

    def return_code(self) -> Optional[int]:
        if self.popen_obj:
            return self.popen_obj.returncode
        else:
            return None

    @staticmethod
    def get_latest_cmo_instance() -> Optional[psutil.Process]:
        latest_inst: Optional[psutil.Process] = None
        for p in psutil.process_iter():
            if p.name() == 'CommandCLI.exe':
                if latest_inst is None:
                    latest_inst = p
                elif p.create_time() > latest_inst.create_time():
                    latest_inst = p
        return latest_inst

    @staticmethod
    def kill_cmo_instance_for_port(port: int) -> None:
        for p in psutil.process_iter():
            if p.name() == 'CommandCLI.exe':
                try:
                    port_arg = p.cmdline().index('-port')
                except Exception:
                    port_arg = None
                try:
                    inst_port = int(p.cmdline()[port_arg + 1]) if port_arg is not None else CMOInstance.CMO_DEFAULT_PORT
                    if inst_port == port:
                        p.kill()
                        return
                except Exception:
                    pass
