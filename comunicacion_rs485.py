"""Control de dos Pico por un bus RS-485 half-duplex."""
import threading
import time
import serial

class ComunicacionRS485:
    def __init__(self, puerto="/dev/ttyUSB0", baudios=9600):
        self.puerto = puerto
        self.lock = threading.Lock()
        self.serial = serial.Serial(puerto, baudios, timeout=1.5, write_timeout=2)
        time.sleep(0.3)

    def comando(self, numero, accion, espera=3):
        if numero not in (1, 2):
            raise ValueError("Modulo invalido")
        solicitud = f"{numero}:{accion}"
        limite = time.monotonic() + espera
        with self.lock:
            self.serial.reset_input_buffer()
            self.serial.write((solicitud + "\n").encode("ascii"))
            self.serial.flush()
            while time.monotonic() < limite:
                restante = limite - time.monotonic()
                self.serial.timeout = max(0.1, min(1.5, restante))
                linea = self.serial.readline()
                if not linea:
                    continue
                texto = linea.decode("utf-8", errors="replace").strip()
                print("RS485", solicitud, "->", texto)
                prefijo = f"{numero}:"
                if texto.startswith(prefijo):
                    return texto[len(prefijo):]
            raise TimeoutError(f"Sin respuesta de Pico {numero} a {accion}")

    def estado(self, numero):
        respuesta = self.comando(numero, "Estado", espera=3)
        if respuesta not in ("Disponible", "Ocupado", "RetireSuBicicleta", "Error"):
            raise RuntimeError(f"Estado inesperado de Pico {numero}: {respuesta}")
        return respuesta

    def cerrar(self):
        self.serial.close()
