import serial as s
import time

import serial.tools.list_ports

class Arduino:
    def __init__(self, baud: int = 9600, timeout: float = 0.1) -> None:
        self.baud: int = baud
        self.timeout: float = timeout
        self._arduino = None
        self.port: str = self.setup_arduino()

    @property
    def arduino(self):
        return self._arduino

    def setup_arduino(self) -> str:
        COMS = [tuple(p) for p in list(serial.tools.list_ports.comports())]
        port: str = ""
        for COM, device, data in COMS:
            if "Arduino" in device:
                port = COM
        if port == "": 
            print("Arduino is not connected to the computer")
            return ""
        else:
            self._arduino = s.Serial(port, self.baud, timeout=self.timeout)
            print(f"Serial connection with the Arduino is setup at {port}")
            time.sleep(1)
            return port

    def write(self, string: str) -> None:
        self.arduino.write(Arduino.encode(string))

    def read(self) -> str:
        string: str = ""
        while string == "":
            string = Arduino.decode(self.arduino.readline())
        return string

    def close(self):
        return self.arduino.close()

    @staticmethod
    def COM_list():
        print([tuple(p) for p in list(serial.tools.list_ports.comports())])

    @staticmethod
    def decode(string: bytes) -> str:
        return str(string, 'utf-8')

    @staticmethod
    def encode(string: str) -> bytes:
        return bytes(string, 'utf-8')
