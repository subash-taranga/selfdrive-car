import serial
import time

from serial.tools.list_ports import comports

for portItem in comports():
    print(portItem)

arduinoSerial = serial.Serial(port='/dev/ttyUSB0', boudrate=115200, timeout=.5)

arduinoSerial.is_open

arduinoSerial.close()

arduinoSerial.open()

numberToSend=423