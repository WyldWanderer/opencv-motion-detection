import serial
import time

class NerfTurret:
    def __init__(self):
        try: 
            self.serial = serial.Serial("COM3", 230400, timeout=1)
            time.sleep(2)
        except serial.SerialException as e:
            print(f"Error: {e}")
            self.serial = None
        self.movement_codes = {
            "right": "5A",
            "left": "08",
            "fire": "16"
        }

    def take_action(self, action):
        if self.serial:
            try: 
                code = f"ir tx NEC:00{self.movement_codes[f"{action}"]}\n"
                self.serial.flushInput()
                self.serial.flushOutput()
                self.serial.write(code.encode())
                self.serial.flush()    
                time.sleep(1)  # Allow time for response

                response = self.serial.readline().decode().strip()
                if response:
                    print(f"Flipper Response: {response}")
            except serial.SerialException as e:
                print(f"Error: {e}")

    def close(self):
        self.serial.close()


nerf = NerfTurret()
nerf.take_action("fire")
time.sleep(2)
nerf.close()

