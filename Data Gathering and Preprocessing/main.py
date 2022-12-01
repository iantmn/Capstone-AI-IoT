from arduino import Arduino
import time

def main():
    arduino = Arduino()
    # time.sleep(.1)
    print(arduino.read())

if __name__ == "__main__":
    main()