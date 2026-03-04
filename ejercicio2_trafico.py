# =============================================================================
# PARCIAL 1 MBA - Ejercicio 2
# Simulación de Tráfico en una Ciudad
# =============================================================================
# Objetivo: Modelar el flujo de vehículos en una ciudad con intersecciones
# y semáforos. Los autos se mueven en una cuadrícula 10x10 representando calles.
# Los semáforos ciclan entre verde, amarillo y rojo controlando el flujo.
# =============================================================================

# Importamos Agent y Model de Mesa para construir el modelo basado en agentes
from mesa import Agent, Model

# Importamos DataCollector para registrar métricas de la simulación en cada paso
from mesa.datacollection import DataCollector

# Importamos matplotlib para la visualización de la simulación
import matplotlib.pyplot as plt

# Importamos matplotlib.patches para dibujar figuras en el grid visual
import matplotlib.patches as mpatches

# Importamos numpy para manejar arrays numéricos en la visualización
import numpy as np

# Importamos random para decisiones aleatorias de los vehículos
import random


# =============================================================================
# CONSTANTES DE CONFIGURACIÓN DEL MODELO
# =============================================================================

# Tamaño de la cuadrícula (10x10 representa la ciudad)
ANCHO_GRID  = 10  # Número de columnas del grid (eje X)
ALTO_GRID   = 10  # Número de filas del grid (eje Y)

# Número de vehículos en la simulación
NUM_AUTOS = 15

# Duración de cada fase del semáforo (en pasos de simulación)
DURACION_VERDE   = 5  # El semáforo permanece en verde durante 5 pasos
DURACION_AMARILLO = 2  # El semáforo permanece en amarillo durante 2 pasos
DURACION_ROJO    = 5  # El semáforo permanece en rojo durante 5 pasos

# Posiciones de las intersecciones donde colocaremos semáforos
# Son las cruces de las calles principales del grid 10x10
POSICIONES_SEMAFOROS = [
    (2, 2),   # Intersección en la esquina inferior-izquierda del centro
    (2, 7),   # Intersección en la esquina superior-izquierda del centro
    (7, 2),   # Intersección en la esquina inferior-derecha del centro
    (7, 7),   # Intersección en la esquina superior-derecha del centro
    (5, 5),   # Intersección central
]

# Posibles direcciones de movimiento de los vehículos (dx, dy)
# Cada tupla representa el desplazamiento en X e Y por paso
DIRECCIONES = [
    (1,  0),  # Moverse a la derecha (Este)
    (-1, 0),  # Moverse a la izquierda (Oeste)
    (0,  1),  # Moverse hacia arriba (Norte)
    (0, -1),  # Moverse hacia abajo (Sur)
]


# =============================================================================
# CLASE AGENTE: SemaforoAgent
# =============================================================================

class SemaforoAgent(Agent):
    """
    Representa un semáforo ubicado en una intersección de la ciudad.
    El semáforo cambia de estado automáticamente según temporizadores:
    Verde -> Amarillo -> Rojo -> Verde -> ...
    """

    def __init__(self, model, posicion, offset_inicial=0):
        """
        Inicializa el semáforo.

        Parámetros:
            model: referencia al modelo de la simulación.
            posicion (tuple): coordenada (x, y) del semáforo en el grid.
            offset_inicial (int): desplazamiento del temporizador inicial para
                                  que no todos los semáforos cambien al mismo tiempo.
        """

        # Llamamos al constructor de la clase padre Agent
        super().__init__(model)

        # Guardamos la posición del semáforo en el grid
        self.posicion = posicion

        # El semáforo comienza en estado "verde" para permitir el flujo inicial
        self.estado = "verde"

        # Inicializamos el temporizador interno con el offset para desfasar semáforos
        # Esto simula que los semáforos de la ciudad no están sincronizados perfectamente
        self.temporizador = offset_inicial

    def step(self):
        """
        Actualiza el estado del semáforo en cada paso de la simulación.
        El ciclo es: verde (5 pasos) -> amarillo (2 pasos) -> rojo (5 pasos) -> repite
        """

        # Incrementamos el temporizador en 1 en cada paso
        self.temporizador += 1

        # Calculamos la duración total de un ciclo completo del semáforo
        ciclo_total = DURACION_VERDE + DURACION_AMARILLO + DURACION_ROJO  # = 12 pasos

        # Calculamos en qué punto del ciclo estamos usando el módulo
        # Esto hace que el ciclo se repita indefinidamente
        fase = self.temporizador % ciclo_total

        # Si estamos en la fase de verde (0 a DURACION_VERDE - 1)...
        if fase < DURACION_VERDE:
            self.estado = "verde"    # El semáforo muestra verde

        # Si estamos en la fase de amarillo (DURACION_VERDE a DURACION_VERDE + DURACION_AMARILLO - 1)...
        elif fase < DURACION_VERDE + DURACION_AMARILLO:
            self.estado = "amarillo" # El semáforo muestra amarillo

        # Si estamos en la fase de rojo (resto del ciclo)...
        else:
            self.estado = "rojo"     # El semáforo muestra rojo

    def esta_en_verde(self):
        """
        Método de consulta: retorna True si el semáforo está en verde.
        Los vehículos usan este método para saber si pueden avanzar.
        """
        # Solo retorna True cuando el estado es exactamente "verde"
        return self.estado == "verde"


# =============================================================================
# CLASE AGENTE: AutoAgent
# =============================================================================

class AutoAgent(Agent):
    """
    Representa un vehículo que se mueve por las calles de la ciudad.
    El auto tiene una dirección de movimiento preferida y avanza en el grid.
    Si encuentra un semáforo en rojo o amarillo, se detiene y espera.
    Si su celda de destino está ocupada, también espera (evita colisiones).
    """

    def __init__(self, model, posicion_inicial, direccion_inicial):
        """
        Inicializa el vehículo.

        Parámetros:
            model: referencia al modelo de la simulación.
            posicion_inicial (tuple): coordenada (x, y) inicial del auto.
            direccion_inicial (tuple): dirección de movimiento inicial (dx, dy).
        """

        # Llamamos al constructor de la clase padre Agent
        super().__init__(model)

        # Guardamos la posición actual del auto en el grid
        self.posicion = posicion_inicial

        # Guardamos la dirección de movimiento actual del auto
        self.direccion = direccion_inicial

        # Estado del auto: "moviéndose" o "detenido"
        self.estado = "moviendose"

        # Contador de pasos que el auto ha estado detenido
        self.pasos_detenido = 0

        # Contador total de movimientos realizados a lo largo de la simulación
        self.movimientos_totales = 0

    def _calcular_posicion_destino(self):
        """
        Calcula la posición a la que el auto intentará moverse.
        Si la celda destino está fuera del grid, el auto cambia de dirección
        y elige una nueva dirección aleatoria (simula giros en la ciudad).

        Retorna:
            tuple: coordenada (x, y) de la posición destino.
        """

        # Calculamos la posición destino sumando la dirección a la posición actual
        nuevo_x = self.posicion[0] + self.direccion[0]  # Nueva coordenada X
        nuevo_y = self.posicion[1] + self.direccion[1]  # Nueva coordenada Y

        # Verificamos si la posición destino está dentro de los límites del grid
        dentro_limites = (
            0 <= nuevo_x < ANCHO_GRID and  # X debe estar entre 0 y ANCHO_GRID-1
            0 <= nuevo_y < ALTO_GRID        # Y debe estar entre 0 y ALTO_GRID-1
        )

        # Si la posición está fuera del grid (el auto llegó al borde de la ciudad)...
        if not dentro_limites:
            # ...el auto cambia de dirección aleatoriamente (como si girara en una esquina)
            self.direccion = random.choice(DIRECCIONES)

            # Recalculamos la posición destino con la nueva dirección
            nuevo_x = self.posicion[0] + self.direccion[0]
            nuevo_y = self.posicion[1] + self.direccion[1]

            # Aseguramos que la nueva posición también esté dentro del grid
            # usando clamp para limitar los valores entre 0 y el tamaño máximo
            nuevo_x = max(0, min(nuevo_x, ANCHO_GRID - 1))
            nuevo_y = max(0, min(nuevo_y, ALTO_GRID - 1))

        # Retornamos la posición destino calculada
        return (nuevo_x, nuevo_y)

    def _hay_semaforo_bloqueante(self, posicion_destino):
        """
        Verifica si hay un semáforo en rojo o amarillo en la posición destino.
        Un semáforo en rojo/amarillo bloquea el paso del vehículo.

        Parámetros:
            posicion_destino (tuple): coordenada (x, y) a verificar.

        Retorna:
            bool: True si hay un semáforo que bloquea, False si puede pasar.
        """

        # Recorremos todos los agentes del modelo que sean semáforos
        for agente in self.model.agents:

            # Verificamos que el agente sea un SemaforoAgent
            if isinstance(agente, SemaforoAgent):

                # Verificamos si el semáforo está en la posición destino del auto
                if agente.posicion == posicion_destino:

                    # Si el semáforo NO está en verde, bloquea el paso
                    if not agente.esta_en_verde():
                        return True  # Hay un semáforo bloqueante

        # Si no encontramos semáforo bloqueante, el auto puede avanzar
        return False

    def _posicion_ocupada_por_auto(self, posicion_destino):
        """
        Verifica si ya hay otro vehículo en la posición destino.
        Dos autos no pueden ocupar la misma celda (evita colisiones).

        Parámetros:
            posicion_destino (tuple): coordenada (x, y) a verificar.

        Retorna:
            bool: True si la posición está ocupada, False si está libre.
        """

        # Recorremos todos los agentes del modelo que sean autos
        for agente in self.model.agents:

            # Nos aseguramos de que sea un AutoAgent y que no sea el auto actual
            if isinstance(agente, AutoAgent) and agente is not self:

                # Si ese otro auto está en la posición destino, está ocupada
                if agente.posicion == posicion_destino:
                    return True  # Posición ocupada

        # Si no encontramos otro auto en esa posición, está libre
        return False

    def step(self):
        """
        Ejecuta el comportamiento del vehículo en cada paso de la simulación:
        1. Calcula a dónde quiere moverse.
        2. Verifica si puede moverse (semáforo, colisión).
        3. Se mueve o se detiene según las condiciones.
        4. Con probabilidad baja, cambia de dirección espontáneamente (giro).
        """

        # Calculamos la posición destino según la dirección actual
        posicion_destino = self._calcular_posicion_destino()

        # Verificamos si hay un semáforo en rojo/amarillo en la posición destino
        bloqueado_por_semaforo = self._hay_semaforo_bloqueante(posicion_destino)

        # Verificamos si hay otro auto en la posición destino
        bloqueado_por_auto = self._posicion_ocupada_por_auto(posicion_destino)

        # Si el auto NO está bloqueado ni por semáforo ni por otro auto...
        if not bloqueado_por_semaforo and not bloqueado_por_auto:

            # ...el auto se mueve a la posición destino
            self.posicion = posicion_destino

            # Actualizamos el estado a "moviéndose"
            self.estado = "moviendose"

            # Reiniciamos el contador de pasos detenido
            self.pasos_detenido = 0

            # Incrementamos el contador de movimientos totales del auto
            self.movimientos_totales += 1

            # Con 15% de probabilidad, el auto cambia de dirección espontáneamente
            # (simula que el conductor decide girar en una intersección)
            if random.random() < 0.15:
                self.direccion = random.choice(DIRECCIONES)

        else:
            # Si está bloqueado, el auto se detiene
            self.estado = "detenido"

            # Incrementamos el contador de cuántos pasos lleva detenido
            self.pasos_detenido += 1

            # Si el auto lleva detenido más de 3 pasos consecutivos...
            if self.pasos_detenido > 3:
                # ...intenta girar a una dirección diferente para buscar otra ruta
                direcciones_alternativas = [d for d in DIRECCIONES if d != self.direccion]

                # Elegimos una dirección alternativa aleatoriamente
                self.direccion = random.choice(direcciones_alternativas)

                # Reiniciamos el contador de pasos detenido tras cambiar de dirección
                self.pasos_detenido = 0


# =============================================================================
# CLASE MODELO: CiudadModel
# =============================================================================

class CiudadModel(Model):
    """
    Modelo principal que representa la ciudad con calles, intersecciones,
    semáforos y vehículos. Gestiona todos los agentes y coordina los pasos.
    """

    def __init__(self, num_autos=NUM_AUTOS, seed=42):
        """
        Inicializa el modelo de la ciudad.

        Parámetros:
            num_autos (int): número de vehículos a simular.
            seed (int): semilla para reproducibilidad.
        """

        # Llamamos al constructor de la clase padre Model de Mesa
        super().__init__(rng=seed)

        # Guardamos el número de autos como atributo
        self.num_autos = num_autos

        # Creamos los semáforos en las intersecciones definidas
        # Cada semáforo recibe un offset diferente para desfasar sus ciclos
        for i, posicion in enumerate(POSICIONES_SEMAFOROS):

            # Creamos un semáforo manualmente (sin create_agents para pasar posición)
            semaforo = SemaforoAgent(self, posicion, offset_inicial=i * 2)

        # Generamos posiciones iniciales aleatorias para los autos
        # Usamos un conjunto para asegurarnos de que no se superpongan
        posiciones_semaforos = set(POSICIONES_SEMAFOROS)  # Posiciones reservadas para semáforos

        # Creamos un conjunto para llevar registro de posiciones ya ocupadas por autos
        posiciones_usadas = set()

        # Creamos cada auto individualmente para asignarle posición y dirección únicas
        for _ in range(num_autos):

            # Buscamos una posición libre para este auto
            while True:
                # Generamos una posición aleatoria en el grid 10x10
                pos_x = random.randint(0, ANCHO_GRID - 1)
                pos_y = random.randint(0, ALTO_GRID - 1)
                pos = (pos_x, pos_y)

                # Verificamos que la posición no esté ocupada por otro auto ni por un semáforo
                if pos not in posiciones_usadas and pos not in posiciones_semaforos:
                    posiciones_usadas.add(pos)  # Marcamos la posición como usada
                    break                        # Salimos del bucle con una posición válida

            # Asignamos una dirección de movimiento inicial aleatoria al auto
            direccion_inicial = random.choice(DIRECCIONES)

            # Creamos el agente auto con la posición y dirección seleccionadas
            auto = AutoAgent(self, pos, direccion_inicial)

        # Configuramos el DataCollector para registrar métricas en cada paso
        self.datacollector = DataCollector(
            model_reporters={
                # Contamos autos que están en movimiento en este paso
                "Autos_moviendose": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, AutoAgent) and a.estado == "moviendose"
                ),
                # Contamos autos que están detenidos en este paso
                "Autos_detenidos": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, AutoAgent) and a.estado == "detenido"
                ),
                # Contamos semáforos en verde en este paso
                "Semaforos_verde": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, SemaforoAgent) and a.estado == "verde"
                ),
                # Contamos semáforos en rojo en este paso
                "Semaforos_rojo": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, SemaforoAgent) and a.estado == "rojo"
                ),
            }
        )

        # Recolectamos los datos iniciales antes de ejecutar pasos
        self.datacollector.collect(self)

    def step(self):
        """
        Ejecuta un paso completo de la simulación:
        1. Primero actualizan los semáforos (para que los autos vean el estado correcto).
        2. Luego se mueven los autos.
        3. Se recolectan los datos del paso.
        """

        # Paso 1: Actualizamos todos los semáforos primero
        # Filtramos solo los SemaforoAgent y los activamos en orden aleatorio
        for agente in list(self.agents):
            if isinstance(agente, SemaforoAgent):
                agente.step()  # El semáforo actualiza su estado (verde/amarillo/rojo)

        # Paso 2: Activamos todos los autos en orden aleatorio
        # shuffle_do garantiza que no haya un auto que siempre tenga ventaja sobre los demás
        autos = [a for a in self.agents if isinstance(a, AutoAgent)]
        random.shuffle(autos)  # Mezclamos el orden de activación de los autos

        # Ejecutamos el step de cada auto
        for auto in autos:
            auto.step()  # El auto decide si moverse o detenerse

        # Paso 3: Recolectamos los datos del modelo tras el movimiento
        self.datacollector.collect(self)


# =============================================================================
# FUNCIÓN DE VISUALIZACIÓN DEL GRID
# =============================================================================

def construir_imagen_grid(modelo):
    """
    Construye una matriz numpy representando el estado visual del grid.
    Cada celda tiene un valor que representa qué hay en ella:
    - 0: celda vacía (calle)
    - 1: auto en movimiento (azul)
    - 2: auto detenido (naranja)
    - 3: semáforo en verde
    - 4: semáforo en amarillo
    - 5: semáforo en rojo

    Parámetros:
        modelo: instancia del CiudadModel.

    Retorna:
        numpy.ndarray: matriz de ALTO_GRID x ANCHO_GRID con valores 0-5.
    """

    # Creamos una matriz de ceros del tamaño del grid (representando calles vacías)
    grid = np.zeros((ALTO_GRID, ANCHO_GRID), dtype=int)

    # Pintamos los semáforos en el grid según su estado actual
    for agente in modelo.agents:
        if isinstance(agente, SemaforoAgent):

            # Obtenemos la posición del semáforo
            x, y = agente.posicion

            # Asignamos el valor correspondiente según el estado del semáforo
            if agente.estado == "verde":
                grid[y][x] = 3   # Verde
            elif agente.estado == "amarillo":
                grid[y][x] = 4   # Amarillo
            else:
                grid[y][x] = 5   # Rojo

    # Pintamos los autos en el grid según su estado (moviéndose o detenido)
    for agente in modelo.agents:
        if isinstance(agente, AutoAgent):

            # Obtenemos la posición del auto
            x, y = agente.posicion

            # Asignamos valor según si el auto está moviéndose o detenido
            if agente.estado == "moviendose":
                grid[y][x] = 1   # Auto en movimiento (azul)
            else:
                grid[y][x] = 2   # Auto detenido (naranja)

    # Retornamos la matriz con el estado visual del grid
    return grid


def visualizar_simulacion(modelo, pasos_datos, num_pasos):
    """
    Genera dos figuras de visualización:
    1. Animación del grid mostrando el movimiento de autos y semáforos.
    2. Gráficas de líneas mostrando la evolución del flujo de tráfico.

    Parámetros:
        modelo: instancia del CiudadModel ya simulado.
        pasos_datos: DataFrame con los datos del DataCollector.
        num_pasos: número total de pasos simulados.
    """

    # ----- FIGURA 1: Estado final del grid -----

    # Creamos una figura para mostrar el estado final de la ciudad
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

    # Definimos el mapa de colores para el grid:
    # 0=gris claro (calle), 1=azul (auto moviéndose), 2=naranja (auto detenido),
    # 3=verde (semáforo verde), 4=amarillo (semáforo amarillo), 5=rojo (semáforo rojo)
    from matplotlib.colors import ListedColormap

    # Creamos un colormap personalizado con 6 colores (uno por cada valor posible)
    colores = ["#d0d0d0", "#2196F3", "#FF9800", "#4CAF50", "#FFEB3B", "#F44336"]
    cmap_grid = ListedColormap(colores)

    # Construimos la imagen del grid con el estado final de la simulación
    imagen_grid = construir_imagen_grid(modelo)

    # Mostramos la imagen del grid usando imshow
    im = ax1.imshow(
        imagen_grid,
        cmap=cmap_grid,        # Colormap personalizado
        vmin=0, vmax=5,        # Rango de valores posibles (0 a 5)
        origin="lower",        # El origen (0,0) está en la esquina inferior-izquierda
        interpolation="nearest" # Sin interpolación para mantener los cuadros nítidos
    )

    # Añadimos la cuadrícula para ver claramente las celdas
    ax1.set_xticks(np.arange(-0.5, ANCHO_GRID, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, ALTO_GRID, 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=0.5)

    # Etiquetas de los ejes
    ax1.set_xlabel("Coordenada X (Este-Oeste)", fontsize=11)
    ax1.set_ylabel("Coordenada Y (Norte-Sur)", fontsize=11)
    ax1.set_title(f"Estado final de la ciudad - Paso {num_pasos}", fontsize=13, fontweight="bold")

    # Creamos una leyenda con los colores usados en el grid
    leyenda_elementos = [
        mpatches.Patch(color="#d0d0d0", label="Calle vacía"),
        mpatches.Patch(color="#2196F3", label="Auto en movimiento"),
        mpatches.Patch(color="#FF9800", label="Auto detenido"),
        mpatches.Patch(color="#4CAF50", label="Semáforo VERDE"),
        mpatches.Patch(color="#FFEB3B", label="Semáforo AMARILLO"),
        mpatches.Patch(color="#F44336", label="Semáforo ROJO"),
    ]
    ax1.legend(handles=leyenda_elementos, loc="upper right", fontsize=9, framealpha=0.9)

    # Ajustamos el layout para que todo se vea bien
    plt.tight_layout()

    # ----- FIGURA 2: Gráficas de flujo de tráfico -----

    # Creamos una figura con dos subplots: flujo de autos y estados de semáforos
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 8))

    # Título general de la segunda figura
    fig2.suptitle(
        "Análisis del Flujo de Tráfico a lo Largo del Tiempo",
        fontsize=14, fontweight="bold"
    )

    # Panel superior: autos en movimiento vs. detenidos
    ax2.plot(
        pasos_datos["Autos_moviendose"],
        label="Autos en movimiento",
        color="#2196F3",       # Azul
        linewidth=2
    )
    ax2.plot(
        pasos_datos["Autos_detenidos"],
        label="Autos detenidos",
        color="#FF9800",       # Naranja
        linewidth=2,
        linestyle="--"
    )

    # Línea de referencia mostrando el total de autos
    ax2.axhline(
        y=NUM_AUTOS,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"Total autos ({NUM_AUTOS})"
    )

    # Configuramos etiquetas y título del panel de autos
    ax2.set_ylabel("Número de autos", fontsize=11)
    ax2.set_title("Flujo de vehículos: en movimiento vs. detenidos", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel inferior: semáforos en verde vs. rojo
    ax3.fill_between(
        range(len(pasos_datos)),
        pasos_datos["Semaforos_verde"],
        alpha=0.4,
        color="#4CAF50",       # Verde
        label="Semáforos en verde"
    )
    ax3.fill_between(
        range(len(pasos_datos)),
        pasos_datos["Semaforos_rojo"],
        alpha=0.4,
        color="#F44336",       # Rojo
        label="Semáforos en rojo"
    )
    ax3.plot(pasos_datos["Semaforos_verde"], color="#4CAF50", linewidth=2)
    ax3.plot(pasos_datos["Semaforos_rojo"],  color="#F44336", linewidth=2)

    # Configuramos etiquetas y título del panel de semáforos
    ax3.set_xlabel("Paso de simulación", fontsize=11)
    ax3.set_ylabel("Número de semáforos", fontsize=11)
    ax3.set_title("Estado de los semáforos a lo largo del tiempo", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Ajustamos el layout de la segunda figura
    plt.tight_layout()

    # Mostramos ambas figuras
    plt.show()


def imprimir_resumen(modelo, pasos_datos, num_pasos):
    """
    Imprime un resumen estadístico de la simulación de tráfico en consola.

    Parámetros:
        modelo: instancia del CiudadModel.
        pasos_datos: DataFrame con los datos recolectados.
        num_pasos: número total de pasos simulados.
    """

    # Calculamos el total de movimientos realizados por todos los autos
    total_movimientos = sum(
        a.movimientos_totales
        for a in modelo.agents
        if isinstance(a, AutoAgent)
    )

    # Calculamos el promedio de autos en movimiento por paso
    promedio_moviendose = pasos_datos["Autos_moviendose"].mean()

    # Calculamos el porcentaje de eficiencia: qué fracción del tiempo los autos se mueven
    porcentaje_eficiencia = (promedio_moviendose / NUM_AUTOS) * 100

    # Imprimimos el encabezado del resumen
    print("\n" + "="*55)
    print("      RESUMEN DE LA SIMULACIÓN - TRÁFICO URBANO")
    print("="*55)

    # Mostramos los parámetros generales
    print(f"  Total de vehículos     : {NUM_AUTOS}")
    print(f"  Semáforos en la ciudad : {len(POSICIONES_SEMAFOROS)}")
    print(f"  Pasos simulados        : {num_pasos}")
    print(f"  Tamaño del grid        : {ANCHO_GRID}x{ALTO_GRID}")
    print()

    # Mostramos métricas del flujo de tráfico
    print("  Métricas del flujo de tráfico:")
    print("  " + "-"*50)
    print(f"  Promedio autos moviéndose : {promedio_moviendose:.1f} / {NUM_AUTOS}")
    print(f"  Eficiencia del flujo      : {porcentaje_eficiencia:.1f}%")
    print(f"  Total movimientos         : {total_movimientos}")
    print()

    # Mostramos el estado final de cada semáforo
    print("  Estado final de semáforos:")
    print("  " + "-"*50)
    for agente in modelo.agents:
        if isinstance(agente, SemaforoAgent):
            # Determinamos el símbolo visual del estado del semáforo
            simbolo = {"verde": "[VERDE]", "amarillo": "[AMARILLO]", "rojo": "[ROJO]"}
            print(f"  Semáforo en {agente.posicion}: {simbolo[agente.estado]}")

    print("="*55 + "\n")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # Definimos el número de pasos de la simulación
    NUM_PASOS = 40

    # Creamos el modelo de la ciudad con 15 autos y semáforos en intersecciones
    modelo = CiudadModel(num_autos=NUM_AUTOS, seed=42)

    # Informamos al usuario que la simulación ha comenzado
    print(f"\nIniciando simulación de tráfico en ciudad {ANCHO_GRID}x{ALTO_GRID}...")
    print(f"Vehículos: {NUM_AUTOS} | Semáforos: {len(POSICIONES_SEMAFOROS)} | Pasos: {NUM_PASOS}")

    # Ejecutamos la simulación paso a paso
    for paso in range(NUM_PASOS):

        # Ejecutamos un paso del modelo (semáforos cambian, autos se mueven)
        modelo.step()

        # Mostramos el progreso cada 10 pasos
        if (paso + 1) % 10 == 0:
            # Contamos autos moviéndose y detenidos en este paso para el log
            moviendose = sum(1 for a in modelo.agents if isinstance(a, AutoAgent) and a.estado == "moviendose")
            detenidos  = sum(1 for a in modelo.agents if isinstance(a, AutoAgent) and a.estado == "detenido")
            print(f"  Paso {paso + 1}/{NUM_PASOS} | Moviéndose: {moviendose} | Detenidos: {detenidos}")

    # Obtenemos el DataFrame con todos los datos recolectados durante la simulación
    datos_simulacion = modelo.datacollector.get_model_vars_dataframe()

    # Imprimimos el resumen estadístico en consola
    imprimir_resumen(modelo, datos_simulacion, NUM_PASOS)

    # Llamamos a la función de visualización para mostrar el grid y las gráficas
    visualizar_simulacion(modelo, datos_simulacion, NUM_PASOS)
