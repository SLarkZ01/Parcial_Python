# =============================================================================
# PARCIAL 1 MBA - Ejercicio 2
# Simulación de Tráfico en una Ciudad con Semáforos
# =============================================================================
# Objetivo: Modelar el flujo de vehículos en una intersección de 4 calles.
# Cada vehículo es un agente que se desplaza en su carril (Norte, Sur, Este u
# Oeste). Dos semáforos coordinados controlan quién puede cruzar la
# intersección: cuando el eje horizontal tiene verde, el vertical tiene rojo
# y viceversa. Los carros respetan la fila: el de atrás no avanza si el de
# adelante está detenido o demasiado cerca.
#
# Estructura del archivo (igual que ejercicio 1):
#   1. Imports
#   2. Constantes globales
#   3. Clase Semaforo  (objeto del modelo, NO es agente Mesa)
#   4. Clase CarroAgent (agente Mesa)
#   5. Clase TraficoModel (modelo Mesa)
#   6. Función animar_simulacion
#   7. Función imprimir_resumen
#   8. Bloque __main__
# =============================================================================


# ── Librería Mesa: base para el modelado basado en agentes ──────────────────
from mesa import Agent, Model                    # Agent: agente base; Model: modelo base
from mesa.datacollection import DataCollector    # Recolector de estadísticas por paso

# ── Matplotlib: visualización y animación ───────────────────────────────────
import matplotlib.pyplot as plt                  # Construcción de la figura
import matplotlib.patches as mpatches            # Rectángulos y parches visuales
from matplotlib.animation import FuncAnimation   # Animación cuadro a cuadro

# ── Utilidades estándar ──────────────────────────────────────────────────────
import random                                    # Decisiones y posiciones aleatorias
import numpy as np                               # Arrays para set_offsets del scatter


# =============================================================================
# CONSTANTES GLOBALES — configuración central de la simulación
# =============================================================================

# ── Semáforos ────────────────────────────────────────────────────────────────

# Número de pasos que un semáforo permanece en verde antes de pasar a amarillo.
# Debe ser >= tiempo de cruce: la intersección mide 0.30 y la velocidad es 0.02,
# por lo que cruzar tarda 0.30 / 0.02 = 15 pasos. Con 35 pasos de verde hay
# margen amplio para que todos los carros en cola crucen cómodamente, y el
# semáforo se percibe visualmente como lento y realista.
CICLO_VERDE = 40

# Número de pasos en amarillo (advertencia: los carros frenan y se detienen).
# 5 pasos da tiempo visual suficiente para notar el cambio de color claramente.
CICLO_AMARILLO = 5

# Número de pasos en rojo (el eje contrario tiene verde durante este tiempo).
# Simétrico con CICLO_VERDE para que ambos ejes reciban el mismo tiempo de paso.
CICLO_ROJO = CICLO_VERDE  # Ambos ejes tienen el mismo tiempo de verde/rojo

# ── Carros ────────────────────────────────────────────────────────────────────

# Avance de cada carro por paso de simulación en coordenadas normalizadas [0, 1].
# Con 0.02 y CICLO_VERDE=20, un carro en STOP_LINE (0.33) puede avanzar
# 20 × 0.02 = 0.40 unidades → llega hasta 0.73, pasando INTER_MAX (0.65) con
# holgura. Esto garantiza que los carros que empiezan a cruzar con verde
# terminen el cruce antes de que el semáforo cambie a rojo.
VELOCIDAD_CARRO = 0.02

# Distancia mínima que debe haber entre dos carros consecutivos del mismo carril.
# Si el carro de adelante está más cerca que esto, el de atrás se detiene.
# Con VELOCIDAD=0.02, una separación de 0.10 equivale a ~5 pasos de distancia,
# suficiente para ver la fila con claridad y evitar solapamiento visual.
DISTANCIA_MIN = 0.10

# Posición (en coordenadas [0, 1]) donde los carros frenan al encontrar semáforo
# en rojo o amarillo. Debe ser ESTRICTAMENTE MENOR que INTER_MIN (0.40) para que
# los carros se detengan ANTES de entrar a la intersección y no se acumulen dentro.
STOP_LINE = 0.33

# Cada cuántos pasos de simulación aparece un nuevo carro en cada extremo.
# Con VELOCIDAD=0.02, un carro tarda 50 pasos en cruzar la pantalla completa.
# Con intervalo 20, los carros se generan lentamente y nunca superan el límite
# de 4 por carril, dando una sensación de tráfico fluido y no saturado.
SPAWN_INTERVALO = 20

# Máximo de carros por carril (por dirección). Se controla individualmente en
# _generar_carro() para que ningún color supere este límite en ningún momento.
MAX_CARROS_POR_CARRIL = 3

# ── Geometría de la intersección ─────────────────────────────────────────────

# Coordenadas del cuadrado central que representa la intersección.
# Se amplía de 0.40-0.60 a 0.35-0.65 (ancho 0.30) para que los dos carriles
# de cada calle tengan espacio suficiente y no se superpongan visualmente.
INTER_MIN = 0.35   # Borde izquierdo/inferior de la intersección
INTER_MAX = 0.65   # Borde derecho/superior de la intersección

# ── Colores de los carros por dirección ─────────────────────────────────────
# Cada dirección tiene un color distintivo para identificarlos visualmente.
COLOR_CARRO = {
    "NORTE": "#E74C3C",   # Rojo: carros que vienen del Sur y van al Norte
    "SUR":   "#3498DB",   # Azul: carros que vienen del Norte y van al Sur
    "ESTE":  "#2ECC71",   # Verde: carros que vienen del Oeste y van al Este
    "OESTE": "#F39C12",   # Naranja: carros que vienen del Este y van al Oeste
}

# ── Estados posibles de un semáforo ─────────────────────────────────────────
VERDE    = "VERDE"
AMARILLO = "AMARILLO"
ROJO     = "ROJO"


# =============================================================================
# CLASE: Semaforo
# =============================================================================

class Semaforo:
    """
    Representa un semáforo que controla un eje de tráfico (horizontal o vertical).

    El ciclo de estados es:
        VERDE (CICLO_VERDE pasos)
          → AMARILLO (CICLO_AMARILLO pasos)
          → ROJO (CICLO_ROJO pasos)
          → VERDE ...

    Dos semáforos están coordinados: cuando uno está en VERDE, el otro está
    en ROJO (o AMARILLO transitando hacia ROJO). La coordinación la maneja
    el modelo, que crea ambos semáforos con fases opuestas.

    Atributos:
        eje (str)           : "HORIZONTAL" o "VERTICAL", solo informativo
        estado (str)        : estado actual (VERDE, AMARILLO o ROJO)
        _contador (int)     : pasos acumulados en el estado actual
        _duracion (int)     : pasos totales que debe durar el estado actual
    """

    def __init__(self, eje: str, estado_inicial: str):
        """
        Inicializa el semáforo.

        Parámetros:
            eje (str)            : "HORIZONTAL" o "VERTICAL"
            estado_inicial (str) : estado con el que arranca (VERDE o ROJO)
        """
        self.eje = eje                        # Eje que controla este semáforo
        self.estado = estado_inicial          # Estado inicial del semáforo
        self._contador = 0                    # Pasos transcurridos en el estado actual
        # Duración inicial según el estado de arranque
        self._duracion = self._duracion_de(estado_inicial)

    def _duracion_de(self, estado: str) -> int:
        """
        Retorna cuántos pasos dura el estado indicado.

        Parámetros:
            estado (str): VERDE, AMARILLO o ROJO

        Retorna:
            int: número de pasos que dura ese estado
        """
        if estado == VERDE:
            return CICLO_VERDE
        elif estado == AMARILLO:
            return CICLO_AMARILLO
        else:  # ROJO
            return CICLO_ROJO

    def _siguiente_estado(self) -> str:
        """
        Retorna el siguiente estado en el ciclo VERDE → AMARILLO → ROJO → VERDE.

        Retorna:
            str: nombre del siguiente estado
        """
        if self.estado == VERDE:
            return AMARILLO    # Después de verde siempre viene amarillo
        elif self.estado == AMARILLO:
            return ROJO        # Después de amarillo siempre viene rojo
        else:
            return VERDE       # Después de rojo vuelve a verde

    def step(self):
        """
        Avanza el semáforo un paso. Si se agota la duración del estado actual,
        transiciona al siguiente estado del ciclo.
        """
        self._contador += 1                          # Contamos un paso más

        if self._contador >= self._duracion:
            # Se acabó el tiempo del estado actual: transicionamos
            self.estado = self._siguiente_estado()   # Nuevo estado
            self._duracion = self._duracion_de(self.estado)  # Duración del nuevo estado
            self._contador = 0                       # Reiniciamos el contador

    def permite_avanzar(self) -> bool:
        """
        Indica si los carros controlados por este semáforo pueden cruzar.
        Solo el estado VERDE permite avanzar; AMARILLO y ROJO detienen el tráfico.

        Retorna:
            bool: True si el semáforo está en VERDE
        """
        return self.estado == VERDE


# =============================================================================
# CLASE: CarroAgent
# =============================================================================

class CarroAgent(Agent):
    """
    Representa un vehículo que circula en una de las 4 direcciones:
    NORTE, SUR, ESTE u OESTE.

    Cada carro tiene:
    - Una dirección fija (no cambia de carril)
    - Una posición normalizada (0.0 = origen, 1.0 = destino / salida)
    - Un estado: MOVIENDO o ESPERANDO

    En cada paso, el carro:
    1. Verifica si el semáforo de su eje lo detiene antes de la intersección
    2. Verifica si el carro de adelante está demasiado cerca (cola)
    3. Si puede avanzar, suma VELOCIDAD_CARRO a su posición
    4. Si supera 1.0 (salió de la pantalla), se marca para eliminación

    Parámetros de inicialización:
        model (TraficoModel) : referencia al modelo
        direccion (str)      : "NORTE", "SUR", "ESTE" u "OESTE"
    """

    def __init__(self, model, direccion: str):
        """
        Crea un carro en el extremo de entrada de su carril.

        Parámetros:
            model (TraficoModel) : modelo al que pertenece
            direccion (str)      : dirección de circulación
        """
        super().__init__(model)                    # Inicializa el agente Mesa

        self.direccion = direccion                 # Dirección del carro (fija)
        self.posicion = 0.0                        # Posición inicial: borde de entrada
        self.estado = "MOVIENDO"                   # Estado inicial: en movimiento
        self.activo = True                         # True mientras está en la pantalla

        # Estado del semáforo en el paso anterior. Se usa para detectar la
        # transición rojo/amarillo → verde y aplicar el arranque escalonado.
        self._semaforo_estado_anterior = None

        # Pasos que aún debe esperar antes de arrancar (arranque escalonado).
        # Cuando el semáforo cambia a verde, cada carro en la cola recibe un
        # retraso proporcional a su posición en la fila: el primero arranca
        # inmediatamente, el segundo espera 2 pasos, el tercero 4, etc.
        self._pasos_espera_arranque = 0

    def _obtener_semaforo(self) -> Semaforo:
        """
        Retorna el semáforo que controla el eje de este carro.
        - NORTE y SUR pertenecen al eje VERTICAL
        - ESTE y OESTE pertenecen al eje HORIZONTAL

        Retorna:
            Semaforo: objeto semáforo que controla a este carro
        """
        if self.direccion in ("NORTE", "SUR"):
            return self.model.semaforo_vertical     # Eje vertical
        else:
            return self.model.semaforo_horizontal   # Eje horizontal

    def _hay_carro_adelante(self) -> bool:
        """
        Comprueba si hay otro carro del mismo carril que esté demasiado cerca
        por delante. Esto implementa la cola: un carro no puede acercarse
        más de DISTANCIA_MIN al que tiene delante.

        Retorna:
            bool: True si debe detenerse por el carro de adelante
        """
        for otro in self.model.agents:
            # Solo comparamos contra carros activos del mismo carril
            if (
                not isinstance(otro, CarroAgent)    # Ignoramos no-carros
                or otro is self                     # Ignoramos a nosotros mismos
                or not otro.activo                  # Ignoramos carros inactivos
                or otro.direccion != self.direccion # Solo mismo carril
            ):
                continue

            # El carro "adelante" tiene mayor posición (más avanzado en el carril)
            diferencia = otro.posicion - self.posicion

            # Si está adelante y dentro de la distancia mínima → hay que detenerse
            if 0 < diferencia < DISTANCIA_MIN:
                return True

        return False  # No hay ningún carro bloqueando

    def step(self):
        """
        Comportamiento del carro en cada paso de simulación.

        Prioridades (en orden):
        1. Si ya salió de la pantalla (posicion > 1.0) → se desactiva
        2. Si el semáforo acaba de cambiar a verde → calcular retraso de arranque
           escalonado según posición en la cola (el primero arranca ya, los de
           atrás esperan 2 pasos por cada carro que tienen delante)
        3. Si aún está en espera de arranque escalonado → descontar un paso
        4. Si el semáforo está en rojo/amarillo Y el carro llegó a STOP_LINE → espera
        5. Si hay un carro adelante dentro de DISTANCIA_MIN → espera (cola)
        6. En cualquier otro caso → avanza VELOCIDAD_CARRO
        """

        # ── Caso 1: el carro ya salió de la pantalla ──────────────────────────
        if self.posicion > 1.0:
            self.activo = False    # Se marca inactivo para que el modelo lo elimine
            self.estado = "FUERA"
            return                 # No hay más acciones que tomar

        # ── Obtener estado actual del semáforo ────────────────────────────────
        semaforo = self._obtener_semaforo()
        estado_actual = semaforo.estado

        # ── Caso 2: el semáforo acaba de cambiar a VERDE ──────────────────────
        # Detectamos la transición comparando con el estado del paso anterior.
        # Solo aplicamos el retraso si el carro está en la zona de cola
        # (entre STOP_LINE e INTER_MIN), es decir, estaba detenido esperando.
        recien_cambio_a_verde = (
            estado_actual == VERDE
            and self._semaforo_estado_anterior != VERDE
            and self.posicion >= STOP_LINE
            and self.posicion <= INTER_MIN
        )

        if recien_cambio_a_verde:
            # Contamos cuántos carros del mismo carril están DELANTE de este
            # y también en la zona de cola (posicion entre STOP_LINE e INTER_MIN).
            # El resultado es la posición en la fila (0 = primero, 1 = segundo, etc.)
            carros_delante_en_cola = sum(
                1 for a in self.model.agents
                if (
                    isinstance(a, CarroAgent)
                    and a is not self
                    and a.activo
                    and a.direccion == self.direccion
                    and a.posicion > self.posicion      # Está delante
                    and a.posicion <= INTER_MIN         # También en la cola
                )
            )
            # Retraso: 2 pasos por cada carro que tiene delante en la cola.
            # El primero (0 carros delante) arranca de inmediato (retraso = 0).
            # El segundo espera 2 pasos, el tercero 4, etc.
            self._pasos_espera_arranque = carros_delante_en_cola * 2

        # Guardamos el estado actual para detectar la transición en el siguiente paso
        self._semaforo_estado_anterior = estado_actual

        # ── Caso 3: aún en espera de arranque escalonado ──────────────────────
        if self._pasos_espera_arranque > 0:
            self._pasos_espera_arranque -= 1   # Descontamos un paso de espera
            self.estado = "ESPERANDO"
            return                             # No avanza todavía

        # ── Caso 4: semáforo en rojo/amarillo y carro en la línea de stop ─────
        # La condición clave: el carro debe estar entre STOP_LINE y INTER_MIN.
        # Esto garantiza que se detenga ANTES de entrar a la intersección.
        # Si ya pasó INTER_MIN (entró a cruzar con verde) no se le detiene,
        # pues interrumpirlo a mitad del cruce causaría acumulación en el centro.
        semaforo_bloqueante = (
            not semaforo.permite_avanzar()           # El semáforo NO está en verde
            and self.posicion >= STOP_LINE           # El carro llegó a la línea de stop
            and self.posicion <= INTER_MIN           # El carro AÚN NO entró a la intersección
        )

        if semaforo_bloqueante:
            self.estado = "ESPERANDO"   # El carro espera en la línea de stop
            return                      # No avanza en este paso

        # ── Caso 5: hay un carro adelante muy cerca (respeto de cola) ─────────
        if self._hay_carro_adelante():
            self.estado = "ESPERANDO"   # El carro espera detrás del de adelante
            return                      # No avanza en este paso

        # ── Caso 6: puede avanzar libremente ──────────────────────────────────
        self.posicion += VELOCIDAD_CARRO    # Mueve el carro hacia adelante
        self.estado = "MOVIENDO"            # El carro está en movimiento


# =============================================================================
# CLASE: TraficoModel
# =============================================================================

class TraficoModel(Model):
    """
    Modelo de tráfico. Gestiona los semáforos, los carros y las estadísticas.

    Responsabilidades:
    - Crear y actualizar los dos semáforos coordinados
    - Generar nuevos carros periódicamente en los bordes de cada carril
    - Eliminar carros que han salido de la pantalla
    - Recolectar datos de conteo de carros activos por dirección
    """

    def __init__(self, seed: int = 42):
        """
        Inicializa el modelo con semáforos y sin carros (se generan en step).

        Parámetros:
            seed (int): semilla para reproducibilidad de la simulación
        """
        super().__init__(rng=seed)   # Constructor Mesa con semilla

        # Paso actual de la simulación (se incrementa en cada step)
        self.paso_actual = 0

        # ── Semáforos coordinados ────────────────────────────────────────────
        # El eje horizontal arranca en VERDE → los carros E-O pueden pasar
        self.semaforo_horizontal = Semaforo("HORIZONTAL", VERDE)

        # El eje vertical arranca en ROJO → los carros N-S deben esperar
        # Para que estén realmente en oposición, iniciamos el vertical con
        # un contador ya avanzado (equivalente a que lleva CICLO_VERDE pasos en rojo)
        self.semaforo_vertical = Semaforo("VERTICAL", ROJO)

        # ── Offsets de spawn por dirección ────────────────────────────────────
        # Cada dirección recibe un desplazamiento aleatorio dentro del intervalo
        # de spawn. Así los carros de cada carril aparecen en pasos distintos
        # en lugar de todos al mismo tiempo, evitando colisiones iniciales.
        self._spawn_offsets = {
            d: random.randint(0, SPAWN_INTERVALO - 1)
            for d in ("NORTE", "SUR", "ESTE", "OESTE")
        }

        # ── DataCollector ─────────────────────────────────────────────────────
        # Registra en cada paso cuántos carros activos hay por dirección
        self.datacollector = DataCollector(
            model_reporters={
                # Carros activos que van hacia el NORTE
                "Norte": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, CarroAgent) and a.activo and a.direccion == "NORTE"
                ),
                # Carros activos que van hacia el SUR
                "Sur": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, CarroAgent) and a.activo and a.direccion == "SUR"
                ),
                # Carros activos que van hacia el ESTE
                "Este": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, CarroAgent) and a.activo and a.direccion == "ESTE"
                ),
                # Carros activos que van hacia el OESTE
                "Oeste": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, CarroAgent) and a.activo and a.direccion == "OESTE"
                ),
                # Semáforo horizontal (para graficar su estado si se desea)
                "Semaforo_H": lambda m: m.semaforo_horizontal.estado,
                # Semáforo vertical
                "Semaforo_V": lambda m: m.semaforo_vertical.estado,
            }
        )

        # Recolectamos datos del estado inicial (sin carros)
        self.datacollector.collect(self)

    def _contar_carros_activos(self) -> int:
        """
        Cuenta cuántos carros activos hay en este momento.

        Retorna:
            int: número de CarroAgent con activo=True
        """
        return sum(
            1 for a in self.agents
            if isinstance(a, CarroAgent) and a.activo
        )

    def _generar_carro(self, direccion: str):
        """
        Intenta crear un nuevo carro en el extremo de entrada del carril indicado.
        No lo crea si:
          - Ya hay un carro cerca del punto de entrada (evita solapamiento al nacer), o
          - El carril ya tiene MAX_CARROS_POR_CARRIL carros activos (evita saturación).

        Parámetros:
            direccion (str): dirección del nuevo carro
        """
        # Contamos cuántos carros activos hay ya en este carril
        carros_en_carril = sum(
            1 for a in self.agents
            if isinstance(a, CarroAgent) and a.activo and a.direccion == direccion
        )

        # Si el carril ya alcanzó el límite por dirección, no generamos más
        if carros_en_carril >= MAX_CARROS_POR_CARRIL:
            return

        # Verificamos que no haya un carro muy cerca del punto de entrada (posicion ≈ 0.0)
        hay_carro_en_entrada = any(
            a for a in self.agents
            if (
                isinstance(a, CarroAgent)
                and a.activo
                and a.direccion == direccion
                and a.posicion < DISTANCIA_MIN  # Otro carro está en la zona de entrada
            )
        )

        if not hay_carro_en_entrada:
            # Creamos el nuevo carro (Mesa le asigna un unique_id automáticamente)
            CarroAgent(self, direccion)

    def step(self):
        """
        Ejecuta un paso completo de la simulación:

        1. Avanzamos el contador de pasos
        2. Actualizamos los semáforos (pueden cambiar de estado)
        3. Generamos nuevos carros en los bordes si toca según el intervalo
        4. Ejecutamos el step() de cada carro activo (en orden de posición,
           del más adelantado al más atrasado, para respetar la cola)
        5. Eliminamos los carros que ya salieron de la pantalla
        6. Recolectamos estadísticas
        """

        # ── 1. Avanzamos el contador global ───────────────────────────────────
        self.paso_actual += 1

        # ── 2. Actualizamos los semáforos ─────────────────────────────────────
        # Ambos semáforos se sincronizan avanzando juntos. La coordinación
        # (uno verde = otro rojo) se mantiene porque se inicializaron en oposición
        # y ambos tienen el mismo CICLO_VERDE y CICLO_ROJO.
        self.semaforo_horizontal.step()
        self.semaforo_vertical.step()

        # ── 3. Generamos nuevos carros si corresponde ─────────────────────────
        # Cada dirección tiene su propio offset aleatorio (_spawn_offsets) para
        # que los carros de distintos carriles aparezcan en pasos diferentes.
        # Esto evita que al inicio todos los carriles generen un carro en el
        # mismo frame. El límite por carril se controla dentro de _generar_carro().
        for direccion in ("NORTE", "SUR", "ESTE", "OESTE"):
            # El offset desplaza el ciclo de spawn de esta dirección de forma
            # exclusiva: (paso + offset) % SPAWN_INTERVALO == 0 ocurre una vez
            # cada SPAWN_INTERVALO pasos, pero en un paso distinto para cada carril.
            offset = self._spawn_offsets[direccion]
            if (self.paso_actual + offset) % SPAWN_INTERVALO == 0:
                self._generar_carro(direccion)   # Un carro nuevo en este carril

        # ── 4. Ejecutamos cada carro en orden (adelante → atrás) ──────────────
        # Ordenar del más adelantado (posicion mayor) al más atrasado garantiza
        # que cuando el de atrás calcula si hay alguien delante, el de adelante
        # ya actualizó su posición. Así la cola funciona correctamente.
        carros_ordenados = sorted(
            [a for a in self.agents if isinstance(a, CarroAgent) and a.activo],
            key=lambda c: c.posicion,
            reverse=True   # Mayor posición primero (el más avanzado)
        )

        for carro in carros_ordenados:
            carro.step()   # Cada carro decide si avanza o espera

        # ── 5. Eliminamos carros inactivos (los que salieron de la pantalla) ───
        carros_inactivos = [
            a for a in self.agents
            if isinstance(a, CarroAgent) and not a.activo
        ]
        for carro in carros_inactivos:
            carro.remove()   # Mesa elimina el agente del modelo

        # ── 6. Recolectamos estadísticas del paso ──────────────────────────────
        self.datacollector.collect(self)


# =============================================================================
# FUNCIÓN DE ANIMACIÓN
# =============================================================================

def animar_simulacion(num_pasos: int = 60, intervalo_ms: int = 120):
    """
    Construye y lanza la animación interactiva de la simulación de tráfico.

    La figura tiene dos paneles:
    - Izquierdo (intersección): vista aérea de la intersección con carros
      moviéndose en sus carriles y semáforos cambiando de color.
    - Derecho (barras): conteo de carros activos por dirección, actualizado
      en cada frame.

    Parámetros:
        num_pasos (int)    : número de pasos/frames de la simulación
        intervalo_ms (int) : milisegundos entre frames (velocidad de la animación)
    """

    # Creamos el modelo (sin carros al inicio; se generan en step)
    modelo = TraficoModel(seed=42)

    # ── Configuración de la figura ────────────────────────────────────────────

    # Figura con dos subplots: intersección (ancho 2) y barras (ancho 1)
    fig, (ax_inter, ax_barras) = plt.subplots(
        1, 2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    # Título general de la figura
    fig.suptitle(
        "Simulación de Tráfico en una Intersección con Semáforos",
        fontsize=15, fontweight="bold", y=0.98
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL IZQUIERDO: Vista aérea de la intersección
    # ─────────────────────────────────────────────────────────────────────────

    ax_inter.set_xlim(0, 1)       # Coordenadas normalizadas [0, 1]
    ax_inter.set_ylim(0, 1)
    ax_inter.set_aspect("equal")  # La intersección debe verse cuadrada
    ax_inter.axis("off")          # Sin ejes numéricos

    # ── Fondo verde (hierba) en las 4 esquinas ───────────────────────────────
    # Las calles ocupan franjas centrales; las esquinas son zonas verdes
    for (x0, y0, ancho, alto) in [
        (0.0,       0.0,       INTER_MIN, INTER_MIN),   # Esquina inferior-izquierda
        (INTER_MAX, 0.0,       1-INTER_MAX, INTER_MIN), # Esquina inferior-derecha
        (0.0,       INTER_MAX, INTER_MIN, 1-INTER_MAX), # Esquina superior-izquierda
        (INTER_MAX, INTER_MAX, 1-INTER_MAX, 1-INTER_MAX), # Esquina superior-derecha
    ]:
        rect_verde = mpatches.Rectangle(
            (x0, y0), ancho, alto,
            facecolor="#4CAF50",   # Verde: hierba
            edgecolor="none",
            zorder=1
        )
        ax_inter.add_patch(rect_verde)

    # ── Calles (franjas grises) ───────────────────────────────────────────────
    # Calle horizontal: franja gris que atraviesa todo el ancho
    calle_h = mpatches.Rectangle(
        (0.0, INTER_MIN),
        1.0, INTER_MAX - INTER_MIN,
        facecolor="#555555",   # Gris oscuro: asfalto
        edgecolor="none",
        zorder=2
    )
    ax_inter.add_patch(calle_h)

    # Calle vertical: franja gris que atraviesa todo el alto
    calle_v = mpatches.Rectangle(
        (INTER_MIN, 0.0),
        INTER_MAX - INTER_MIN, 1.0,
        facecolor="#555555",
        edgecolor="none",
        zorder=2
    )
    ax_inter.add_patch(calle_v)

    # ── Líneas de carril (divisorias amarillas punteadas) ────────────────────
    # Línea central horizontal (divide carril Este del carril Oeste)
    ax_inter.plot(
        [0.0, INTER_MIN], [0.5, 0.5],        # Tramo izquierdo de la calle horizontal
        color="#FFD700", linewidth=1.5, linestyle="--", zorder=3
    )
    ax_inter.plot(
        [INTER_MAX, 1.0], [0.5, 0.5],        # Tramo derecho de la calle horizontal
        color="#FFD700", linewidth=1.5, linestyle="--", zorder=3
    )

    # Línea central vertical (divide carril Norte del carril Sur)
    ax_inter.plot(
        [0.5, 0.5], [0.0, INTER_MIN],        # Tramo inferior de la calle vertical
        color="#FFD700", linewidth=1.5, linestyle="--", zorder=3
    )
    ax_inter.plot(
        [0.5, 0.5], [INTER_MAX, 1.0],        # Tramo superior de la calle vertical
        color="#FFD700", linewidth=1.5, linestyle="--", zorder=3
    )

    # ── Zona de intersección (cuadrado central) ───────────────────────────────
    inter_rect = mpatches.Rectangle(
        (INTER_MIN, INTER_MIN),
        INTER_MAX - INTER_MIN, INTER_MAX - INTER_MIN,
        facecolor="#666666",   # Gris un poco más claro que las calles
        edgecolor="none",
        zorder=3
    )
    ax_inter.add_patch(inter_rect)

    # Paso de cebra horizontal (líneas blancas en los bordes de la intersección)
    num_rayas = 5
    ancho_raya = (INTER_MAX - INTER_MIN) / (num_rayas * 2 - 1)
    for i in range(num_rayas):
        x_raya = INTER_MIN + i * ancho_raya * 2    # Posición X de cada raya
        # Cebra en el borde izquierdo de la intersección (para el carril vertical)
        cebra = mpatches.Rectangle(
            (x_raya, INTER_MIN),
            ancho_raya, INTER_MAX - INTER_MIN,
            facecolor="white", alpha=0.25, edgecolor="none", zorder=4
        )
        ax_inter.add_patch(cebra)

    # ── Etiquetas de dirección en los extremos de cada carril ────────────────
    etiquetas_dir = [
        # (x, y, texto, color)
        (0.5, 0.97, "↓ SUR",   COLOR_CARRO["SUR"]),    # Carros van hacia abajo
        (0.5, 0.03, "↑ NORTE", COLOR_CARRO["NORTE"]),  # Carros van hacia arriba
        (0.03, 0.5, "→ ESTE",  COLOR_CARRO["ESTE"]),   # Carros van hacia la derecha
        (0.97, 0.5, "← OESTE", COLOR_CARRO["OESTE"]),  # Carros van hacia la izquierda
    ]
    for (x, y, texto, color) in etiquetas_dir:
        ax_inter.text(
            x, y, texto,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=color,
            zorder=6
        )

    # ── Semáforos visuales (rectángulo negro con círculo de color) ───────────
    # Con INTER_MIN=0.35, las cajas de semáforo se ubican justo en el borde de
    # la intersección para que no queden sobre la hierba ni sobre los carriles.

    # Caja del semáforo horizontal: esquina izquierda del borde de la intersección,
    # centrada verticalmente en la calle horizontal (y = 0.5)
    caja_semaforo_h = mpatches.FancyBboxPatch(
        (INTER_MIN - 0.065, 0.5 - 0.025), 0.055, 0.05,
        boxstyle="round,pad=0.005",
        facecolor="#1a1a1a", edgecolor="#333333", linewidth=1,
        zorder=7
    )
    ax_inter.add_patch(caja_semaforo_h)

    # Círculo interior del semáforo horizontal (el color cambia con el estado)
    circulo_semaforo_h = mpatches.Circle(
        (INTER_MIN - 0.037, 0.5), 0.016,
        facecolor="green",   # Inicia en verde
        edgecolor="none",
        zorder=8
    )
    ax_inter.add_patch(circulo_semaforo_h)

    # Etiqueta "E-O" sobre la caja del semáforo horizontal
    ax_inter.text(
        INTER_MIN - 0.037, 0.5 + 0.038,
        "E-O", ha="center", va="bottom",
        fontsize=7, color="white", zorder=8
    )

    # Caja del semáforo vertical: borde inferior de la intersección,
    # centrada horizontalmente en la calle vertical (x = 0.5)
    caja_semaforo_v = mpatches.FancyBboxPatch(
        (0.5 - 0.025, INTER_MIN - 0.065), 0.05, 0.055,
        boxstyle="round,pad=0.005",
        facecolor="#1a1a1a", edgecolor="#333333", linewidth=1,
        zorder=7
    )
    ax_inter.add_patch(caja_semaforo_v)

    # Círculo interior del semáforo vertical
    circulo_semaforo_v = mpatches.Circle(
        (0.5, INTER_MIN - 0.037), 0.016,
        facecolor="red",    # Inicia en rojo
        edgecolor="none",
        zorder=8
    )
    ax_inter.add_patch(circulo_semaforo_v)

    # Etiqueta "N-S" bajo la caja del semáforo vertical
    ax_inter.text(
        0.5, INTER_MIN - 0.072,
        "N-S", ha="center", va="top",
        fontsize=7, color="white", zorder=8
    )

    # ── Scatter de carros ────────────────────────────────────────────────────
    # Usaremos scatter para dibujar los carros como puntos de colores.
    # Se inicializa con un array NumPy de forma (0, 2): esto garantiza que
    # matplotlib tenga la estructura 2D correcta desde el primer frame,
    # evitando el IndexError que ocurre cuando se pasa una lista vacía []
    # y matplotlib intenta hacer offsets[:, 0] sobre un array 1D de shape (0,).
    scatter_carros = ax_inter.scatter(
        np.empty((0, 2))[:, 0], np.empty((0, 2))[:, 1],  # Array vacío con forma correcta
        s=120,          # Tamaño reducido: los carros son más pequeños que el carril
        marker="s",     # Cuadrado (s = square) para simular un carro desde arriba
        zorder=9,
        edgecolors="#1a1a1a",
        linewidths=0.8
    )

    # ── Texto del paso actual ────────────────────────────────────────────────
    texto_paso = ax_inter.text(
        0.02, 0.02,
        "Paso: 0",
        transform=ax_inter.transAxes,
        fontsize=11, color="white", fontweight="bold", zorder=10
    )

    # ── Texto del estado de los semáforos ────────────────────────────────────
    texto_semaforos = ax_inter.text(
        0.5, 0.02,
        "",
        transform=ax_inter.transAxes,
        fontsize=9, color="white",
        ha="center", va="bottom", zorder=10
    )

    # ── Texto de conteo de carros ────────────────────────────────────────────
    texto_conteo = ax_inter.text(
        0.5, 0.97,
        "",
        transform=ax_inter.transAxes,
        fontsize=9, color="white",
        ha="center", va="top", zorder=10
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL DERECHO: Barras de conteo por dirección
    # ─────────────────────────────────────────────────────────────────────────

    nombres_dir = ["Norte", "Sur", "Este", "Oeste"]    # Etiquetas del eje X
    x_pos = list(range(4))                             # Posiciones de las barras [0,1,2,3]
    colores_barras = [
        COLOR_CARRO["NORTE"],   # Rojo para Norte
        COLOR_CARRO["SUR"],     # Azul para Sur
        COLOR_CARRO["ESTE"],    # Verde para Este
        COLOR_CARRO["OESTE"],   # Naranja para Oeste
    ]

    # Barras inicializadas en 0 (se actualizan en cada frame)
    barras = ax_barras.bar(
        x_pos, [0, 0, 0, 0],
        color=colores_barras,
        alpha=0.85, zorder=2
    )

    # Configuración del eje de barras
    ax_barras.set_xticks(x_pos)
    ax_barras.set_xticklabels(nombres_dir, fontsize=11)
    ax_barras.set_ylim(0, MAX_CARROS_POR_CARRIL + 2)   # Máximo por dirección + margen
    ax_barras.set_ylabel("Carros activos", fontsize=11)
    ax_barras.set_title("Carros activos por dirección", fontsize=11, fontweight="bold")
    ax_barras.grid(True, axis="y", alpha=0.3, zorder=0)
    ax_barras.set_facecolor("#FAFAFA")

    # Etiquetas numéricas encima de cada barra (se actualizan cada frame)
    etiquetas_barras = []
    for i in range(4):
        etq = ax_barras.text(
            x_pos[i], 0, "",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#2C3E50"
        )
        etiquetas_barras.append(etq)

    # Texto del estado de los semáforos en el panel de barras
    texto_semaforo_barras = ax_barras.text(
        0.5, 0.97, "",
        transform=ax_barras.transAxes,
        fontsize=9, ha="center", va="top",
        color="#333333"
    )

    # Ajuste de márgenes
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ─────────────────────────────────────────────────────────────────────────
    # Mapa de colores de semáforo para el círculo visual
    # ─────────────────────────────────────────────────────────────────────────
    COLOR_SEMAFORO = {
        VERDE:    "#00CC44",   # Verde brillante
        AMARILLO: "#FFD700",   # Amarillo
        ROJO:     "#FF3333",   # Rojo brillante
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Función de actualización por frame
    # ─────────────────────────────────────────────────────────────────────────

    def actualizar(frame):
        """
        Se ejecuta una vez por frame. Avanza el modelo un paso y actualiza
        todos los elementos visuales: scatter de carros, semáforos y barras.

        Parámetro:
            frame (int): número del frame actual (0 = estado inicial)
        """

        # El frame 0 muestra el estado inicial sin ejecutar el modelo
        if frame > 0:
            modelo.step()   # Avanza la simulación un paso

        # ── Posiciones y colores de los carros activos ────────────────────────
        xs = []    # Coordenadas X de cada carro en el canvas
        ys = []    # Coordenadas Y de cada carro en el canvas
        cols = []  # Color de cada carro según su dirección

        for carro in modelo.agents:
            if not isinstance(carro, CarroAgent) or not carro.activo:
                continue   # Solo procesamos carros activos

            # Convertimos la posición lógica [0, 1] del carro a coordenadas
            # visuales (x, y) en el canvas según la dirección de circulación.
            #
            # Con INTER_MIN=0.35 e INTER_MAX=0.65, la calle tiene ancho 0.30.
            # El centro del eje es 0.50. Cada carril se desplaza ±0.09 del centro,
            # lo que da 0.18 de separación visual entre carriles opuestos (holgada).
            #
            # Eje VERTICAL (x = 0.50):
            #   Carril SUR   → x = 0.59  (derecha del centro, va hacia abajo)
            #   Carril NORTE → x = 0.41  (izquierda del centro, va hacia arriba)
            #
            # Eje HORIZONTAL (y = 0.50):
            #   Carril ESTE  → y = 0.59  (arriba del centro, va hacia la derecha)
            #   Carril OESTE → y = 0.41  (abajo del centro, va hacia la izquierda)

            if carro.direccion == "SUR":
                # Va de Norte a Sur: entra por arriba (y=1) y sale por abajo (y=0)
                # Carril derecho del eje vertical (x ligeramente mayor que 0.5)
                x = 0.59                             # Carril derecho, separado del Norte
                y = 1.0 - carro.posicion             # Posición invertida: entra por arriba

            elif carro.direccion == "NORTE":
                # Va de Sur a Norte: entra por abajo (y=0) y sale por arriba (y=1)
                # Carril izquierdo del eje vertical (x ligeramente menor que 0.5)
                x = 0.41                             # Carril izquierdo, separado del Sur
                y = carro.posicion                   # Posición directa: entra por abajo

            elif carro.direccion == "ESTE":
                # Va de Oeste a Este: entra por la izquierda (x=0) y sale por la derecha
                # Carril superior del eje horizontal (y ligeramente mayor que 0.5)
                x = carro.posicion                   # Posición directa: entra por izquierda
                y = 0.59                             # Carril superior, separado del Oeste

            else:  # "OESTE"
                # Va de Este a Oeste: entra por la derecha (x=1) y sale por la izquierda
                # Carril inferior del eje horizontal (y ligeramente menor que 0.5)
                x = 1.0 - carro.posicion             # Posición invertida: entra por derecha
                y = 0.41                             # Carril inferior, separado del Este

            xs.append(x)
            ys.append(y)
            cols.append(COLOR_CARRO[carro.direccion])

        # Actualizamos el scatter con las posiciones y colores nuevos.
        # Se usa np.column_stack para construir un array de forma (N, 2) que
        # matplotlib requiere internamente en set_offsets. Cuando no hay carros
        # se pasa np.empty((0, 2)) para mantener la forma 2D y evitar el
        # IndexError que ocurre si se pasa una lista vacía [] (array 1D shape (0,)).
        if xs:
            # Hay carros: construimos el array (N, 2) con columnas X e Y
            offsets = np.column_stack([xs, ys])
        else:
            # Sin carros: array vacío pero con la forma 2D correcta
            offsets = np.empty((0, 2))

        scatter_carros.set_offsets(offsets)

        # Actualizamos los colores; set_color acepta lista vacía sin problema
        scatter_carros.set_color(cols)

        # ── Actualizamos el color de los círculos de semáforo ─────────────────
        estado_h = modelo.semaforo_horizontal.estado   # Estado actual del eje horizontal
        estado_v = modelo.semaforo_vertical.estado     # Estado actual del eje vertical

        circulo_semaforo_h.set_facecolor(COLOR_SEMAFORO[estado_h])
        circulo_semaforo_v.set_facecolor(COLOR_SEMAFORO[estado_v])

        # ── Textos informativos en el panel de intersección ───────────────────
        texto_paso.set_text(f"Paso: {frame}  /  {num_pasos}")

        texto_semaforos.set_text(
            f"E-O: {estado_h}   |   N-S: {estado_v}"
        )

        # Contamos carros activos por dirección para el texto
        n_norte = sum(
            1 for a in modelo.agents
            if isinstance(a, CarroAgent) and a.activo and a.direccion == "NORTE"
        )
        n_sur = sum(
            1 for a in modelo.agents
            if isinstance(a, CarroAgent) and a.activo and a.direccion == "SUR"
        )
        n_este = sum(
            1 for a in modelo.agents
            if isinstance(a, CarroAgent) and a.activo and a.direccion == "ESTE"
        )
        n_oeste = sum(
            1 for a in modelo.agents
            if isinstance(a, CarroAgent) and a.activo and a.direccion == "OESTE"
        )

        texto_conteo.set_text(
            f"↑N:{n_norte}  ↓S:{n_sur}  →E:{n_este}  ←O:{n_oeste}"
            f"  (Total: {n_norte+n_sur+n_este+n_oeste})"
        )

        # ── Actualizamos las barras del panel derecho ─────────────────────────
        valores = [n_norte, n_sur, n_este, n_oeste]
        for i, (barra, valor) in enumerate(zip(barras, valores)):
            barra.set_height(valor)                         # Nueva altura
            etiquetas_barras[i].set_text(str(valor))        # Número encima
            etiquetas_barras[i].set_y(valor + 0.1)          # Posición del número

        # Texto del estado de semáforos en el panel de barras
        texto_semaforo_barras.set_text(
            f"Semáforo E-O: {estado_h}   |   Semáforo N-S: {estado_v}"
        )

        # Retornamos los artistas modificados para que matplotlib los redibuje
        return (
            [scatter_carros, texto_paso, texto_semaforos, texto_conteo,
             circulo_semaforo_h, circulo_semaforo_v, texto_semaforo_barras]
            + list(barras)
            + etiquetas_barras
        )

    # ── Creamos y lanzamos la animación ───────────────────────────────────────
    # FuncAnimation llama a actualizar(frame) para cada frame de 0 a num_pasos
    anim = FuncAnimation(
        fig,
        actualizar,
        frames=num_pasos + 1,    # +1 para incluir el frame 0 (estado inicial)
        interval=intervalo_ms,   # Milisegundos entre frames
        repeat=True,             # Repite la animación al terminar
        blit=False               # blit=False para compatibilidad en Windows
    )

    # Mostramos la ventana con la animación
    plt.show()

    # Retornamos el modelo para acceder a datos si se quiere el resumen
    return modelo


# =============================================================================
# FUNCIÓN DE RESUMEN
# =============================================================================

def imprimir_resumen(modelo: TraficoModel, num_pasos: int):
    """
    Imprime estadísticas finales de la simulación en consola.
    Se muestra después de cerrar la ventana de animación.

    Parámetros:
        modelo (TraficoModel) : el modelo ya ejecutado
        num_pasos (int)       : número de pasos simulados
    """
    # Obtenemos el DataFrame con los datos recolectados por paso
    datos = modelo.datacollector.get_model_vars_dataframe()

    print("\n" + "=" * 58)
    print("   RESUMEN DE LA SIMULACIÓN - TRÁFICO EN INTERSECCIÓN")
    print("=" * 58)
    print(f"  Pasos simulados       : {num_pasos}")
    print(f"  Velocidad de carros   : {VELOCIDAD_CARRO} unidades/paso")
    print(f"  Ciclo verde           : {CICLO_VERDE} pasos")
    print(f"  Ciclo amarillo        : {CICLO_AMARILLO} pasos")
    print(f"  Intervalo de spawn    : cada {SPAWN_INTERVALO} pasos")
    print(f"  Máx. carros por carril: {MAX_CARROS_POR_CARRIL}")
    print()
    print("  Estadísticas de carros activos por dirección:")
    print("  " + "-" * 52)
    print(f"  {'Dirección':<12} {'Promedio':>10} {'Máximo':>10} {'Final':>10}")
    print("  " + "-" * 52)

    for col in ["Norte", "Sur", "Este", "Oeste"]:
        promedio = datos[col].mean()      # Media de carros activos en esa dirección
        maximo   = datos[col].max()       # Pico máximo de carros simultáneos
        final    = datos[col].iloc[-1]    # Carros activos en el último paso
        print(f"  {col:<12} {promedio:>10.1f} {maximo:>10} {final:>10}")

    print("=" * 58 + "\n")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # ── Parámetros de la simulación ───────────────────────────────────────────
    NUM_PASOS    = 120   # Número de pasos/frames de la animación
    INTERVALO_MS = 120   # Milisegundos entre frames (120ms = ~8fps, ritmo fluido)

    print("\n" + "=" * 58)
    print("  SIMULACIÓN DE TRÁFICO EN UNA INTERSECCIÓN")
    print("=" * 58)
    print(f"  Pasos           : {NUM_PASOS}")
    print(f"  Intervalo       : {INTERVALO_MS} ms por frame")
    print(f"  Ciclo semáforo  : {CICLO_VERDE}v / {CICLO_AMARILLO}a / {CICLO_ROJO}r pasos")
    print(f"  Spawn cada      : {SPAWN_INTERVALO} pasos por dirección")
    print(f"  Máx. carros por carril: {MAX_CARROS_POR_CARRIL}")
    print("  Cierra la ventana para ver el resumen en consola.")
    print("=" * 58)

    # ── Lanzamos la animación ─────────────────────────────────────────────────
    modelo_final = animar_simulacion(
        num_pasos=NUM_PASOS,
        intervalo_ms=INTERVALO_MS
    )

    # ── Resumen estadístico tras cerrar la ventana ────────────────────────────
    imprimir_resumen(modelo_final, NUM_PASOS)
