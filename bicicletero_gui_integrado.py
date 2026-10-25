import cv2
import numpy as np
import time
import os
import spidev
import gpiod

from gpiod.line import Direction, Value
from ai_edge_litert.interpreter import Interpreter


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

CAMERA = "/dev/video0"

MODELO = (
    "Face_Recognition_Orange_Pi_3B/"
    "mobilefacenet.tflite"
)

MODELO_DETECTOR = (
    "/usr/share/opencv4/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

ARCHIVO_ESTADO = "bicicletero_estado.npz"

ANCHO_CAMARA = 640
ALTO_CAMARA = 480

ANCHO_LCD = 320
ALTO_LCD = 240

UMBRAL = 0.65
CAPTURAS_VALIDAS = 3
TIEMPO_CAPTURA = 12

FPS_LCD = 8


# ============================================================
# CONFIGURACION TOUCH
# ============================================================

RAW_X_MIN = 600
RAW_X_MAX = 3600

RAW_Y_MIN = 350
RAW_Y_MAX = 3600

GPIO_CHIP = "/dev/gpiochip0"

GPIO_DC = 22
GPIO_RST = 27
GPIO_IRQ = 17


# ============================================================
# DETECTOR FACIAL
# ============================================================

print("Cargando detector facial...")

detector = cv2.CascadeClassifier(
    MODELO_DETECTOR
)

if detector.empty():
    raise RuntimeError(
        "No se pudo cargar Haar Cascade"
    )

print("Detector facial OK")


# ============================================================
# MOBILEFACENET
# ============================================================

print("Cargando MobileFaceNet...")

interpreter = Interpreter(
    model_path=MODELO
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("MobileFaceNet OK")


# ============================================================
# GPIO
# ============================================================

gpio_lcd = gpiod.request_lines(
    GPIO_CHIP,
    consumer="bicicletero-lcd",
    config={
        GPIO_DC: gpiod.LineSettings(
            direction=Direction.OUTPUT,
            output_value=Value.INACTIVE
        ),

        GPIO_RST: gpiod.LineSettings(
            direction=Direction.OUTPUT,
            output_value=Value.ACTIVE
        ),
    },
)


gpio_touch = gpiod.request_lines(
    GPIO_CHIP,
    consumer="bicicletero-touch",
    config={
        GPIO_IRQ: gpiod.LineSettings(
            direction=Direction.INPUT
        )
    },
)


# ============================================================
# SPI LCD
# ============================================================

lcd = spidev.SpiDev()
lcd.open(0, 0)
lcd.max_speed_hz = 32_000_000
lcd.mode = 0


# ============================================================
# SPI TOUCH
# ============================================================

touch = spidev.SpiDev()
touch.open(0, 1)
touch.max_speed_hz = 1_000_000
touch.mode = 0


# ============================================================
# FUNCIONES GPIO
# ============================================================

def dc_bajo():
    gpio_lcd.set_value(
        GPIO_DC,
        Value.INACTIVE
    )


def dc_alto():
    gpio_lcd.set_value(
        GPIO_DC,
        Value.ACTIVE
    )


def rst_bajo():
    gpio_lcd.set_value(
        GPIO_RST,
        Value.INACTIVE
    )


def rst_alto():
    gpio_lcd.set_value(
        GPIO_RST,
        Value.ACTIVE
    )


# ============================================================
# ILI9341
# ============================================================

def comando(valor):

    dc_bajo()
    lcd.writebytes([valor])


def datos(valores):

    dc_alto()
    lcd.writebytes(list(valores))


def reset_lcd():

    rst_alto()
    time.sleep(0.05)

    rst_bajo()
    time.sleep(0.10)

    rst_alto()
    time.sleep(0.15)


def inicializar_lcd():

    reset_lcd()

    comando(0x01)
    time.sleep(0.15)

    comando(0x28)

    comando(0xC0)
    datos([0x23])

    comando(0xC1)
    datos([0x10])

    comando(0xC5)
    datos([0x3E, 0x28])

    comando(0xC7)
    datos([0x86])

    comando(0x36)
    datos([0x28])

    comando(0x3A)
    datos([0x55])

    comando(0xB1)
    datos([0x00, 0x18])

    comando(0xB6)
    datos([0x08, 0x82, 0x27])

    comando(0x26)
    datos([0x01])

    comando(0x11)
    time.sleep(0.15)

    comando(0x29)
    time.sleep(0.10)


def ventana(x0, y0, x1, y1):

    comando(0x2A)

    datos([
        x0 >> 8,
        x0 & 0xFF,
        x1 >> 8,
        x1 & 0xFF
    ])

    comando(0x2B)

    datos([
        y0 >> 8,
        y0 & 0xFF,
        y1 >> 8,
        y1 & 0xFF
    ])

    comando(0x2C)


# ============================================================
# MOSTRAR IMAGEN EN LCD
# ============================================================

def enviar_imagen(imagen):

    if imagen.shape[:2] != (
        ALTO_LCD,
        ANCHO_LCD
    ):

        imagen = cv2.resize(
            imagen,
            (ANCHO_LCD, ALTO_LCD),
            interpolation=cv2.INTER_AREA
        )

    b = imagen[:, :, 0].astype(np.uint16)
    g = imagen[:, :, 1].astype(np.uint16)
    r = imagen[:, :, 2].astype(np.uint16)

    rgb565 = (
        ((r & 0xF8) << 8)
        |
        ((g & 0xFC) << 3)
        |
        (b >> 3)
    )

    buffer = np.empty(
        (ALTO_LCD, ANCHO_LCD, 2),
        dtype=np.uint8
    )

    buffer[:, :, 0] = (
        rgb565 >> 8
    ).astype(np.uint8)

    buffer[:, :, 1] = (
        rgb565 & 0xFF
    ).astype(np.uint8)

    ventana(
        0,
        0,
        ANCHO_LCD - 1,
        ALTO_LCD - 1
    )

    dc_alto()

    buffer = buffer.reshape(-1)

    BLOQUE = 4096

    for inicio in range(
        0,
        len(buffer),
        BLOQUE
    ):

        lcd.writebytes2(
            buffer[
                inicio:
                inicio + BLOQUE
            ]
        )


# ============================================================
# TOUCH
# ============================================================

def leer_adc(comando_touch):

    r = touch.xfer2([
        comando_touch,
        0x00,
        0x00
    ])

    return (
        ((r[1] << 8) | r[2]) >> 3
    ) & 0x0FFF


def limitar(valor, minimo, maximo):

    return max(
        minimo,
        min(valor, maximo)
    )


def convertir_touch(raw_x, raw_y):

    x = int(
        (raw_y - RAW_Y_MIN)
        * 319
        / (RAW_Y_MAX - RAW_Y_MIN)
    )

    y = int(
        (RAW_X_MAX - raw_x)
        * 239
        / (RAW_X_MAX - RAW_X_MIN)
    )

    x = limitar(x, 0, 319)
    y = limitar(y, 0, 239)

    return x, y


def touch_presionado():

    return (
        gpio_touch.get_value(GPIO_IRQ)
        == Value.INACTIVE
    )


def leer_posicion_touch():

    muestras_x = []
    muestras_y = []

    for _ in range(5):

        muestras_x.append(
            leer_adc(0xD0)
        )

        muestras_y.append(
            leer_adc(0x90)
        )

    raw_x = int(
        np.median(muestras_x)
    )

    raw_y = int(
        np.median(muestras_y)
    )

    return convertir_touch(
        raw_x,
        raw_y
    )


def esperar_soltar_touch():

    while touch_presionado():
        time.sleep(0.01)

    time.sleep(0.05)


# ============================================================
# EMBEDDING
# ============================================================

def obtener_embedding(rostro):

    rostro = cv2.resize(
        rostro,
        (112, 112),
        interpolation=cv2.INTER_AREA
    )

    rostro = cv2.cvtColor(
        rostro,
        cv2.COLOR_BGR2RGB
    )

    rostro = rostro.astype(
        np.float32
    )

    rostro = (
        rostro - 127.5
    ) / 128.0

    rostro = np.expand_dims(
        rostro,
        axis=0
    )

    interpreter.set_tensor(
        input_index,
        rostro
    )

    interpreter.invoke()

    embedding = interpreter.get_tensor(
        output_index
    )[0]

    norma = np.linalg.norm(
        embedding
    )

    if norma > 0:
        embedding = embedding / norma

    return embedding.astype(
        np.float32
    )


# ============================================================
# DETECCION FACIAL
# ============================================================

def detectar_rostro(frame):

    gris = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gris = cv2.equalizeHist(gris)

    rostros = detector.detectMultiScale(
        gris,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(rostros) != 1:
        return None, None

    x, y, w, h = rostros[0]

    margen_x = int(w * 0.15)
    margen_y = int(h * 0.15)

    x1 = max(
        0,
        x - margen_x
    )

    y1 = max(
        0,
        y - margen_y
    )

    x2 = min(
        frame.shape[1],
        x + w + margen_x
    )

    y2 = min(
        frame.shape[0],
        y + h + margen_y
    )

    rostro = frame[
        y1:y2,
        x1:x2
    ]

    if rostro.size == 0:
        return None, None

    return rostro, (
        x,
        y,
        w,
        h
    )


# ============================================================
# ESTADO POSICIONES
# ============================================================

posiciones = {
    1: None,
    2: None
}


def guardar_estado():

    datos_estado = {}

    for numero, embedding in posiciones.items():

        if embedding is not None:

            datos_estado[
                f"posicion{numero}"
            ] = embedding

    np.savez(
        ARCHIVO_ESTADO,
        **datos_estado
    )


def cargar_estado():

    if not os.path.exists(
        ARCHIVO_ESTADO
    ):
        return

    try:

        datos_estado = np.load(
            ARCHIVO_ESTADO
        )

        for numero in posiciones:

            clave = (
                f"posicion{numero}"
            )

            if clave in datos_estado.files:

                posiciones[numero] = (
                    datos_estado[clave]
                    .astype(np.float32)
                )

    except Exception as e:

        print(
            "Error cargando estado:",
            e
        )


# ============================================================
# PANTALLA PRINCIPAL
# ============================================================

def pantalla_principal():

    imagen = np.zeros(
        (ALTO_LCD, ANCHO_LCD, 3),
        dtype=np.uint8
    )

    imagen[:] = (35, 35, 35)

    cv2.putText(
        imagen,
        "BICICLETERO SMART",
        (42, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    dibujar_modulo(
        imagen,
        1,
        15,
        55,
        150,
        175
    )

    dibujar_modulo(
        imagen,
        2,
        170,
        55,
        305,
        175
    )

    cv2.putText(
        imagen,
        "Seleccione una posicion",
        (58, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    enviar_imagen(imagen)


def dibujar_modulo(
    imagen,
    numero,
    x1,
    y1,
    x2,
    y2
):

    ocupado = (
        posiciones[numero]
        is not None
    )

    if ocupado:

        color = (
            0,
            0,
            190
        )

        texto_estado = "OCUPADA"

    else:

        color = (
            0,
            170,
            0
        )

        texto_estado = "LIBRE"

    cv2.rectangle(
        imagen,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    cv2.rectangle(
        imagen,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    cv2.putText(
        imagen,
        f"MODULO {numero}",
        (x1 + 18, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )

    cv2.putText(
        imagen,
        texto_estado,
        (x1 + 25, y1 + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.putText(
        imagen,
        "TOCAR",
        (x1 + 34, y1 + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


# ============================================================
# PANTALLAS DE MENSAJES
# ============================================================

def pantalla_mensaje(
    linea1,
    linea2="",
    color=(255, 255, 255)
):

    imagen = np.zeros(
        (ALTO_LCD, ANCHO_LCD, 3),
        dtype=np.uint8
    )

    imagen[:] = (
        35,
        35,
        35
    )

    cv2.putText(
        imagen,
        linea1,
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )

    if linea2:

        cv2.putText(
            imagen,
            linea2,
            (25, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    enviar_imagen(imagen)


# ============================================================
# CAPTURA CON VISTA PREVIA
# ============================================================

def capturar_embedding(modo):

    cap = cv2.VideoCapture(
        CAMERA,
        cv2.CAP_V4L2
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        ANCHO_CAMARA
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        ALTO_CAMARA
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        pantalla_mensaje(
            "ERROR CAMARA"
        )

        time.sleep(2)

        return None

    time.sleep(0.5)

    for _ in range(5):
        cap.read()

    embeddings = []

    inicio = time.time()

    ultimo_envio_lcd = 0

    try:

        while (
            time.time() - inicio
            < TIEMPO_CAPTURA
        ):

            ret, frame = cap.read()

            if not ret:
                continue

            # ------------------------------------------------
            # DETECCION SOBRE FRAME ORIGINAL
            # ------------------------------------------------

            rostro, caja = detectar_rostro(
                frame
            )

            # ------------------------------------------------
            # FRAME SOLO PARA VISUALIZACION
            # ------------------------------------------------

            vista = cv2.resize(
                frame,
                (
                    ANCHO_LCD,
                    ALTO_LCD
                ),
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------
            # RECTANGULO DE ROSTRO
            # ------------------------------------------------

            if caja is not None:

                x, y, w, h = caja

                # Coordenadas 640x480 -> 320x240
                x = int(
                    x * ANCHO_LCD
                    / ANCHO_CAMARA
                )

                y = int(
                    y * ALTO_LCD
                    / ALTO_CAMARA
                )

                w = int(
                    w * ANCHO_LCD
                    / ANCHO_CAMARA
                )

                h = int(
                    h * ALTO_LCD
                    / ALTO_CAMARA
                )

             
                cv2.rectangle(
                    vista,
                    (x, y),
                    (
                        x + w,
                        y + h
                    ),
                    (0, 255, 0),
                    2
                )

            # ------------------------------------------------
            # BARRA INFERIOR
            # ------------------------------------------------

            cv2.rectangle(
                vista,
                (0, 190),
                (319, 239),
                (0, 0, 0),
                -1
            )

            if modo == "registro":

                texto = "REGISTRANDO"

            else:

                texto = "VERIFICANDO"

            cv2.putText(
                vista,
                texto,
                (8, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2
            )

            cv2.putText(
                vista,
                (
                    f"Capturas: "
                    f"{len(embeddings)}/"
                    f"{CAPTURAS_VALIDAS}"
                ),
                (8, 232),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            # ------------------------------------------------
            # ACTUALIZAR LCD MAXIMO 8 FPS
            # ------------------------------------------------

            ahora = time.time()

            if (
                ahora - ultimo_envio_lcd
                >= 1.0 / FPS_LCD
            ):

                enviar_imagen(vista)

                ultimo_envio_lcd = ahora

            # ------------------------------------------------
            # CAPTURA FACIAL
            # ------------------------------------------------

            if rostro is None:
                continue

            embedding = obtener_embedding(
                rostro
            )

            embeddings.append(
                embedding
            )

            print(
                "Captura valida",
                len(embeddings),
                "/",
                CAPTURAS_VALIDAS
            )

            if (
                len(embeddings)
                >= CAPTURAS_VALIDAS
            ):
                break

            # Separar las muestras
            time.sleep(0.35)

    finally:

        cap.release()

    if (
        len(embeddings)
        < CAPTURAS_VALIDAS
    ):

        pantalla_mensaje(
            "NO SE DETECTO",
            "Intente nuevamente",
            (0, 0, 255)
        )

        time.sleep(2)

        return None

    embedding_final = np.mean(
        embeddings,
        axis=0
    )

    norma = np.linalg.norm(
        embedding_final
    )

    if norma > 0:

        embedding_final = (
            embedding_final
            / norma
        )

    return embedding_final.astype(
        np.float32
    )


# ============================================================
# OCUPAR POSICION
# ============================================================

def ocupar_posicion(numero):

    pantalla_mensaje(
        f"MODULO {numero}",
        "Mire hacia la camara"
    )

    time.sleep(0.7)

    embedding = capturar_embedding(
        "registro"
    )

    if embedding is None:
        return

    posiciones[numero] = embedding

    guardar_estado()

    pantalla_mensaje(
        "REGISTRO OK",
        f"Modulo {numero} ocupado",
        (0, 255, 0)
    )

    print(
        f"Modulo {numero}: "
        f"usuario registrado"
    )

    # FUTURO:
    # rs485_guardar(numero)

    time.sleep(2)


# ============================================================
# LIBERAR POSICION
# ============================================================

def intentar_liberar(numero):

    pantalla_mensaje(
        f"MODULO {numero}",
        "Verificando usuario"
    )

    time.sleep(0.7)

    nuevo_embedding = (
        capturar_embedding(
            "verificacion"
        )
    )

    if nuevo_embedding is None:
        return

    referencia = posiciones[numero]

    similitud = float(
        np.dot(
            nuevo_embedding,
            referencia
        )
    )

    print(
        f"Similitud modulo {numero}: "
        f"{similitud:.4f}"
    )

    if similitud >= UMBRAL:

        pantalla_mensaje(
            "AUTORIZADO",
            f"Similitud {similitud:.2f}",
            (0, 255, 0)
        )

        print(
            f"Modulo {numero}: "
            f"usuario autorizado"
        )

        # FUTURO:
        # Aquí irá:
        #
        # rs485_retirar(numero)
        #
        # Y NO liberaremos la posición
        # hasta recibir confirmación del Pico.

        posiciones[numero] = None

        guardar_estado()

        time.sleep(2)

    else:

        pantalla_mensaje(
            "ACCESO DENEGADO",
            f"Similitud {similitud:.2f}",
            (0, 0, 255)
        )

        print(
            f"Modulo {numero}: "
            f"acceso denegado"
        )

        time.sleep(2)


# ============================================================
# PROCESAR MODULO
# ============================================================

def procesar_modulo(numero):

    if posiciones[numero] is None:

        ocupar_posicion(numero)

    else:

        intentar_liberar(numero)

    pantalla_principal()


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

try:

    print()
    print("==============================")
    print("BICICLETERO SMART")
    print("==============================")
    print()

    inicializar_lcd()

    cargar_estado()

    pantalla_principal()

    print("Sistema iniciado.")
    print("Ctrl+C para terminar.")
    print()

    while True:

        if not touch_presionado():

            time.sleep(0.01)
            continue

        # Debounce
        time.sleep(0.03)

        if not touch_presionado():
            continue

        x, y = leer_posicion_touch()

        print(
            f"Touch X={x} Y={y}"
        )

        esperar_soltar_touch()

        # MODULO 1
        if (
            15 <= x <= 150
            and
            55 <= y <= 175
        ):

            procesar_modulo(1)

        # MODULO 2
        elif (
            170 <= x <= 305
            and
            55 <= y <= 175
        ):

            procesar_modulo(2)


except KeyboardInterrupt:

    print()
    print("Cerrando sistema...")


except Exception as e:

    print()
    print("ERROR:")
    print(e)


finally:

    touch.close()
    lcd.close()

    gpio_touch.release()
    gpio_lcd.release()

    print("Recursos liberados.")



