# =============================================================================
# PARCIAL 1 MBA - Ejercicio 1
# Simulación del Movimiento de Estudiantes en la Universidad
# =============================================================================
# Objetivo: Modelar el comportamiento de estudiantes en un campus universitario,
# simulando su movimiento entre el aula, la biblioteca y la cafetería.
# Cada estudiante tiene un ciclo de actividad: pasa un tiempo en un espacio
# y luego decide moverse a otro. Si su espacio favorito está lleno, busca otro.
# La visualización es una ANIMACIÓN en tiempo real: cada punto representa un
# estudiante y se ve cómo se redistribuyen entre las zonas en cada paso.
# =============================================================================

# Importamos Agent y Model de la librería Mesa para construir el modelo ABM
from mesa import Agent, Model

# Importamos DataCollector para registrar estadísticas en cada paso
from mesa.datacollection import DataCollector

# Importamos pyplot para construir la figura de la animación
import matplotlib.pyplot as plt

# Importamos FuncAnimation para animar la figura cuadro a cuadro
from matplotlib.animation import FuncAnimation

# Importamos patches para dibujar los rectángulos de las zonas
import matplotlib.patches as mpatches

# Importamos random para decisiones aleatorias de los agentes
import random

# Importamos numpy para operaciones de transformación sobre el path SVG
import numpy as np

# Importamos svgpath2mpl para convertir el path SVG en un marcador de matplotlib
import svgpath2mpl

# Importamos matplotlib.path y matplotlib.transforms para centrar y escalar el marcador SVG
from matplotlib.path import Path
import matplotlib.transforms as mtransforms


# =============================================================================
# CARGA DEL ÍCONO SVG DE PERSONA COMO MARCADOR PERSONALIZADO
# =============================================================================

# Cadena del path SVG extraída del archivo persona.svg (viewBox 0 0 256 256).
# Este path dibuja la silueta de una persona (cabeza + cuerpo + brazos).
_SVG_PATH_STR = (
    "M127.8877,84a36,36,0,1,0-36-36A36.04062,36.04062,0,0,0,127.8877,84Zm0-48"
    "a12,12,0,1,1-12,12A12.01344,12.01344,0,0,1,127.8877,36Z"
    "M234.29,138.17383A11.99491,11.99491,0,0,1,217.82617,142.29"
    "C217.5,142.09668,185.4541,123.37305,140,120.407v27.03015"
    "l60.96875,68.59015a12.00007,12.00007,0,0,1-17.9375,15.94532"
    "L128,170.0625,72.96875,231.97266a12.00007,12.00007,0,0,1-17.9375-15.94532"
    "L116,147.43719V120.40234"
    "C70.2832,123.35205,38.51611,142.0849,38.17383,142.29"
    "A12,12,0,1,1,25.82617,121.71"
    "C27.57568,120.66016,69.35059,96,128,96"
    "s100.42432,24.66016,102.17383,25.71"
    "A12.00021,12.00021,0,0,1,234.29,138.17383Z"
)

# Convertimos el string del path SVG en un objeto Path de matplotlib
_raw_path = svgpath2mpl.parse_path(_SVG_PATH_STR)

# Obtenemos los vértices del path para calcular su bounding box
_verts = _raw_path.vertices.copy()  # Array Nx2 con las coordenadas de los vértices

# Calculamos el centro del bounding box del path para centrarlo en el origen
_x_min, _y_min = _verts.min(axis=0)  # Esquina inferior izquierda del bounding box
_x_max, _y_max = _verts.max(axis=0)  # Esquina superior derecha del bounding box
_cx = (_x_min + _x_max) / 2.0        # Centro X del bounding box
_cy = (_y_min + _y_max) / 2.0        # Centro Y del bounding box

# Desplazamos todos los vértices para que el centro quede en (0, 0)
_verts_centered = _verts - np.array([_cx, _cy])

# Calculamos el factor de escala para normalizar el path a un tamaño de ±0.5
_scale = max(_x_max - _x_min, _y_max - _y_min)  # Extensión máxima del path
_verts_norm = _verts_centered / _scale            # Vértices normalizados en [-0.5, 0.5]

# En SVG el eje Y apunta hacia abajo; en matplotlib hacia arriba.
# Invertimos el eje Y para que la figura no quede de cabeza.
_verts_norm[:, 1] *= -1.0

# Construimos el marcador final como Path de matplotlib con los vértices normalizados
PERSONA_MARKER = Path(_verts_norm, _raw_path.codes)  # Path listo para usar en scatter


# =============================================================================
# DEFINICIÓN DE ESPACIOS DEL CAMPUS
# =============================================================================

# Cada espacio tiene capacidad máxima, zona visual en el canvas y color de fondo.
# La zona visual usa coordenadas en el rango [0, 1] de matplotlib.
ESPACIOS = {
    "Aula": {
        "capacidad": 30,          # Máximo de estudiantes simultáneos en el aula
        "zona_x": (0.02, 0.30),   # Rango horizontal de la zona en el canvas
        "zona_y": (0.05, 0.90),   # Rango vertical de la zona en el canvas
        "color":  "#AED6F1",      # Fondo azul claro para el aula
    },
    "Biblioteca": {
        "capacidad": 15,          # Máximo de estudiantes en la biblioteca
        "zona_x": (0.36, 0.64),   # Zona central del canvas
        "zona_y": (0.05, 0.90),
        "color":  "#A9DFBF",      # Fondo verde claro para la biblioteca
    },
    "Cafeteria": {
        "capacidad": 20,          # Máximo de estudiantes en la cafetería
        "zona_x": (0.70, 0.98),   # Zona derecha del canvas
        "zona_y": (0.05, 0.90),
        "color":  "#FAD7A0",      # Fondo naranja claro para la cafetería
    },
}

# Umbral de ocupación (80%): si un espacio supera este porcentaje de su
# capacidad, se considera lleno y los estudiantes buscarán otro lugar
UMBRAL_OCUPACION = 0.80

# Colores de los puntos de los estudiantes según su ubicación actual
COLORES_ESPACIO = {
    "Aula":       "#1A5276",  # Azul oscuro: estudiantes en el aula
    "Biblioteca": "#1E8449",  # Verde oscuro: estudiantes en biblioteca
    "Cafeteria":  "#D35400",  # Naranja oscuro: estudiantes en cafetería
}

# Probabilidad base de que un estudiante decida cambiar de espacio en cada paso,
# aunque su espacio actual tenga capacidad disponible (simula comportamiento libre)
PROB_CAMBIO_VOLUNTARIO = 0.20  # 20% de probabilidad de moverse por decisión propia

# Duración mínima y máxima (en pasos) que un estudiante permanece en un espacio
# antes de considerar moverse por ciclo de actividad
DURACION_MIN = 2   # Al menos 2 pasos en el mismo lugar
DURACION_MAX = 6   # Como máximo 6 pasos antes de querer moverse


# =============================================================================
# CLASE AGENTE: EstudianteAgent
# =============================================================================

class EstudianteAgent(Agent):
    """
    Representa a un estudiante que se mueve entre los espacios del campus.
    Cada agente tiene:
    - Una ubicación lógica ("Aula", "Biblioteca", "Cafeteria")
    - Una posición visual (px, py) dentro de su zona en el canvas
    - Un temporizador de permanencia: cuántos pasos lleva en su espacio actual
    - Una duración objetivo: cuántos pasos quiere quedarse antes de moverse
    """

    def __init__(self, model):
        # Inicializamos el agente con Mesa (asigna unique_id automáticamente)
        super().__init__(model)

        # Todos los estudiantes comienzan en el Aula al inicio de la simulación
        self.ubicacion = "Aula"

        # Generamos una posición visual aleatoria dentro de la zona del Aula
        self.px, self.py = self._posicion_aleatoria_en_zona("Aula")

        # Contador de pasos que lleva el estudiante en su espacio actual
        self.pasos_en_espacio = 0

        # Duración objetivo: cuántos pasos quiere permanecer (aleatoria entre MIN y MAX)
        self.duracion_objetivo = random.randint(DURACION_MIN, DURACION_MAX)

        # Historial de ubicaciones para estadísticas
        self.historial = ["Aula"]

    def _posicion_aleatoria_en_zona(self, nombre_espacio):
        """
        Genera coordenadas (x, y) aleatorias dentro de la zona visual del espacio.
        Se usa tanto al inicio como cada vez que el estudiante se mueve.

        Parámetros:
            nombre_espacio (str): nombre del espacio destino

        Retorna:
            tuple (float, float): coordenadas (px, py) en el canvas [0, 1]
        """
        espacio = ESPACIOS[nombre_espacio]

        # Generamos X aleatoria dentro del rango horizontal de la zona (con margen)
        px = random.uniform(espacio["zona_x"][0] + 0.02, espacio["zona_x"][1] - 0.02)

        # Generamos Y aleatoria dentro del rango vertical (con margen para el título)
        py = random.uniform(espacio["zona_y"][0] + 0.08, espacio["zona_y"][1] - 0.04)

        return px, py

    def _contar_ocupacion(self, nombre_espacio):
        """
        Cuenta cuántos estudiantes hay actualmente en el espacio indicado.

        Parámetros:
            nombre_espacio (str): espacio a contar

        Retorna:
            int: número de EstudianteAgent con esa ubicación
        """
        return sum(
            1 for a in self.model.agents
            if isinstance(a, EstudianteAgent) and a.ubicacion == nombre_espacio
        )

    def _espacio_tiene_cupo(self, nombre_espacio):
        """
        Verifica si el espacio indicado tiene capacidad disponible (< 80%).

        Parámetros:
            nombre_espacio (str): espacio a verificar

        Retorna:
            bool: True si hay cupo, False si está lleno
        """
        ocupacion = self._contar_ocupacion(nombre_espacio)
        capacidad = ESPACIOS[nombre_espacio]["capacidad"]
        return (ocupacion / capacidad) < UMBRAL_OCUPACION

    def _moverse_a(self, destino):
        """
        Mueve al estudiante al espacio destino:
        - Actualiza su ubicación lógica
        - Genera nueva posición visual dentro de la zona destino
        - Reinicia el temporizador de permanencia
        - Asigna nueva duración objetivo aleatoria

        Parámetros:
            destino (str): nombre del espacio destino
        """
        self.ubicacion = destino                                  # Cambia ubicación lógica
        self.px, self.py = self._posicion_aleatoria_en_zona(destino)  # Nueva pos. visual
        self.pasos_en_espacio = 0                                 # Reinicia el contador
        self.duracion_objetivo = random.randint(DURACION_MIN, DURACION_MAX)  # Nueva duración

    def step(self):
        """
        Comportamiento del estudiante en cada paso de la simulación.

        Lógica de decisión (en orden de prioridad):
        1. Si el espacio actual está LLENO (>80%) → busca otro espacio con cupo
        2. Si cumplió su duración objetivo → con 80% de prob. se mueve a otro lugar
        3. Si no cumplió la duración → con 20% de prob. se mueve voluntariamente
        4. Si decide moverse pero todos los espacios están llenos → se queda
        """

        # Incrementamos el contador de pasos en el espacio actual
        self.pasos_en_espacio += 1

        # --- Caso 1: el espacio actual está demasiado lleno ---
        # Si la ocupación supera el umbral, el estudiante DEBE buscar otro lugar
        ocupacion_actual = self._contar_ocupacion(self.ubicacion)
        capacidad_actual = ESPACIOS[self.ubicacion]["capacidad"]
        espacio_lleno = (ocupacion_actual / capacidad_actual) > UMBRAL_OCUPACION

        # --- Caso 2: el estudiante cumplió su tiempo en el espacio ---
        # Si lleva suficientes pasos, quiere moverse con alta probabilidad (80%)
        cumplio_duracion = (
            self.pasos_en_espacio >= self.duracion_objetivo
            and random.random() < 0.80
        )

        # --- Caso 3: cambio voluntario aunque no haya cumplido la duración ---
        # Con 20% de probabilidad el estudiante decide moverse por su cuenta
        cambio_voluntario = random.random() < PROB_CAMBIO_VOLUNTARIO

        # Decidimos si el estudiante intentará moverse en este paso
        intentar_moverse = espacio_lleno or cumplio_duracion or cambio_voluntario

        if intentar_moverse:
            # Construimos la lista de destinos posibles distintos al actual
            destinos_posibles = [e for e in ESPACIOS.keys() if e != self.ubicacion]

            # Ordenamos los destinos: primero los que tienen cupo disponible
            # (esto garantiza que el estudiante prefiera espacios con espacio libre)
            destinos_posibles.sort(
                key=lambda e: self._contar_ocupacion(e) / ESPACIOS[e]["capacidad"]
            )

            # Intentamos movernos al destino con menor ocupación que tenga cupo
            for destino in destinos_posibles:
                if self._espacio_tiene_cupo(destino):
                    self._moverse_a(destino)  # Movemos al estudiante al destino
                    break                     # Solo nos movemos una vez por paso

        # Registramos la ubicación actual en el historial (para estadísticas)
        self.historial.append(self.ubicacion)


# =============================================================================
# CLASE MODELO: UniversidadModel
# =============================================================================

class UniversidadModel(Model):
    """
    Modelo del campus universitario. Gestiona todos los estudiantes y
    recolecta estadísticas de ocupación por espacio en cada paso.
    """

    def __init__(self, num_estudiantes=50, seed=42):
        """
        Inicializa el modelo con los agentes y el recolector de datos.

        Parámetros:
            num_estudiantes (int): número de estudiantes a simular
            seed (int): semilla para reproducibilidad
        """
        # Constructor de la clase padre de Mesa
        super().__init__(rng=seed)

        # Guardamos el número de estudiantes
        self.num_estudiantes = num_estudiantes

        # Creamos todos los agentes estudiantes de una sola vez con Mesa
        EstudianteAgent.create_agents(self, num_estudiantes)

        # Configuramos el DataCollector para registrar ocupación por espacio
        self.datacollector = DataCollector(
            model_reporters={
                # Número de estudiantes en el Aula en cada paso
                "Aula": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, EstudianteAgent) and a.ubicacion == "Aula"
                ),
                # Número de estudiantes en la Biblioteca en cada paso
                "Biblioteca": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, EstudianteAgent) and a.ubicacion == "Biblioteca"
                ),
                # Número de estudiantes en la Cafetería en cada paso
                "Cafeteria": lambda m: sum(
                    1 for a in m.agents
                    if isinstance(a, EstudianteAgent) and a.ubicacion == "Cafeteria"
                ),
            }
        )

        # Recolectamos datos del estado inicial (paso 0: todos en el Aula)
        self.datacollector.collect(self)

    def step(self):
        """
        Ejecuta un paso de la simulación:
        1. Activa todos los agentes en orden aleatorio (evita sesgos de orden)
        2. Recolecta datos de ocupación tras el movimiento
        """
        # Activa cada agente en orden aleatorio
        self.agents.shuffle_do("step")

        # Registra las ocupaciones resultantes
        self.datacollector.collect(self)


# =============================================================================
# FUNCIÓN DE ANIMACIÓN
# =============================================================================

def animar_simulacion(num_pasos=30, num_estudiantes=50, intervalo_ms=700):
    """
    Construye y lanza la animación interactiva de la simulación.

    La figura tiene dos paneles:
    - Izquierdo (campus): puntos de colores representando estudiantes en sus zonas.
      Los puntos SE MUEVEN entre zonas en cada frame.
    - Derecho (barras): gráfica de barras con la ocupación actual, se actualiza
      en cada frame junto con los puntos.

    Parámetros:
        num_pasos (int): número de pasos de simulación (frames de la animación)
        num_estudiantes (int): número de estudiantes
        intervalo_ms (int): milisegundos entre frames (700 = ritmo legible)
    """

    # Creamos el modelo; los agentes se inicializan todos en el Aula
    modelo = UniversidadModel(num_estudiantes=num_estudiantes, seed=42)

    # ---- Configuración de la figura ----

    # Figura con dos subplots: campus (ratio 2) y barras (ratio 1)
    fig, (ax_campus, ax_barras) = plt.subplots(
        1, 2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    # Título general de la figura
    fig.suptitle(
        "Simulacion del Movimiento de Estudiantes en la Universidad",
        fontsize=15, fontweight="bold", y=0.98
    )

    # ---- Panel del campus (izquierdo) ----

    ax_campus.set_xlim(0, 1)       # El canvas usa coordenadas normalizadas [0, 1]
    ax_campus.set_ylim(0, 1)
    ax_campus.axis("off")          # Sin ejes numéricos
    ax_campus.set_facecolor("#F4F6F7")  # Fondo gris muy claro (representa el exterior)

    # Dibujamos el rectángulo de fondo de cada zona del campus
    for nombre, datos in ESPACIOS.items():
        ancho = datos["zona_x"][1] - datos["zona_x"][0]
        alto  = datos["zona_y"][1] - datos["zona_y"][0]

        # Rectángulo con bordes redondeados para cada zona
        rect = mpatches.FancyBboxPatch(
            (datos["zona_x"][0], datos["zona_y"][0]),
            ancho, alto,
            boxstyle="round,pad=0.01",
            linewidth=2,
            edgecolor="#555555",
            facecolor=datos["color"],
            alpha=0.55,
            zorder=1
        )
        ax_campus.add_patch(rect)

        # Nombre del espacio en la parte superior de cada zona
        ax_campus.text(
            (datos["zona_x"][0] + datos["zona_x"][1]) / 2,
            datos["zona_y"][1] - 0.03,
            nombre,
            ha="center", va="top",
            fontsize=13, fontweight="bold", color="#2C3E50", zorder=3
        )

        # Capacidad máxima bajo el nombre
        ax_campus.text(
            (datos["zona_x"][0] + datos["zona_x"][1]) / 2,
            datos["zona_y"][1] - 0.09,
            f"Cap. max: {datos['capacidad']}",
            ha="center", va="top",
            fontsize=9, color="#566573", zorder=3
        )

    # Scatter vacío: los íconos de persona se añaden en el primer frame.
    # marker=PERSONA_MARKER usa el path SVG normalizado como símbolo del agente.
    # s=220 controla el tamaño del ícono en puntos² (más grande = más visible).
    # zorder=5 hace que los íconos queden sobre los rectángulos de zona.
    scatter = ax_campus.scatter(
        [], [], s=220,
        marker=PERSONA_MARKER,
        zorder=5,
        edgecolors="white",
        linewidths=0.5
    )

    # Texto del número de paso en la esquina inferior izquierda
    texto_paso = ax_campus.text(
        0.02, 0.01,
        "Paso: 0 / " + str(num_pasos),
        transform=ax_campus.transAxes,
        fontsize=12, color="#2C3E50", fontweight="bold"
    )

    # Texto de conteo por espacio en la parte superior del canvas
    texto_conteo = ax_campus.text(
        0.5, 0.97, "",
        transform=ax_campus.transAxes,
        fontsize=10, color="#333333",
        ha="center", va="top"
    )

    # ---- Panel de barras (derecho) ----

    nombres_espacios = list(ESPACIOS.keys())   # ["Aula", "Biblioteca", "Cafeteria"]
    x_pos = list(range(len(nombres_espacios))) # [0, 1, 2]
    colores_barras = ["#2196F3", "#4CAF50", "#FF9800"]  # Azul, verde, naranja

    # Barras de ocupación inicializadas en 0 (se actualizan cada frame)
    barras = ax_barras.bar(
        x_pos, [0, 0, 0],
        color=colores_barras,
        alpha=0.85, zorder=2
    )

    # Líneas punteadas de capacidad máxima (fijas, no cambian con la animación)
    for i, nombre in enumerate(nombres_espacios):
        cap = ESPACIOS[nombre]["capacidad"]
        ax_barras.axhline(
            y=cap,
            xmin=i / len(nombres_espacios) + 0.04,
            xmax=(i + 1) / len(nombres_espacios) - 0.04,
            color=colores_barras[i],
            linestyle="--", linewidth=2, alpha=0.6, zorder=3
        )

    # Etiquetas del eje X con nombres de los espacios
    ax_barras.set_xticks(x_pos)
    ax_barras.set_xticklabels(nombres_espacios, fontsize=11)

    # Fijamos el límite del eje Y al total de estudiantes para escala estable
    ax_barras.set_ylim(0, num_estudiantes + 2)
    ax_barras.set_ylabel("Numero de estudiantes", fontsize=11)
    ax_barras.set_title("Ocupacion actual por espacio", fontsize=11, fontweight="bold")
    ax_barras.grid(True, axis="y", alpha=0.3, zorder=0)
    ax_barras.set_facecolor("#FAFAFA")

    # Etiquetas numéricas encima de cada barra (se actualizan cada frame)
    etiquetas_barras = []
    for i in range(len(nombres_espacios)):
        etq = ax_barras.text(
            x_pos[i], 0, "",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#2C3E50"
        )
        etiquetas_barras.append(etq)

    # Leyenda del panel de barras explicando las líneas de capacidad
    leyenda_cap = mpatches.Patch(
        facecolor="none", edgecolor="gray",
        linestyle="--", linewidth=2,
        label="--- Capacidad maxima"
    )
    ax_barras.legend(handles=[leyenda_cap], fontsize=9, loc="upper right")

    # Ajuste de márgenes de la figura completa
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ---- Función de actualización de cada frame ----

    def actualizar(frame):
        """
        Se llama una vez por frame. Ejecuta un paso del modelo y actualiza
        el scatter (posiciones y colores de los puntos) y las barras.

        frame=0 muestra el estado inicial sin ejecutar el modelo.
        frame>=1 ejecuta modelo.step() y luego actualiza los gráficos.
        """

        # Ejecutamos un paso de la simulación (excepto en el frame inicial)
        if frame > 0:
            modelo.step()  # Los agentes deciden si moverse y actualizan px, py

        # --- Recopilamos posiciones y colores actuales de todos los agentes ---
        xs = []       # Coordenadas X visuales de cada estudiante
        ys = []       # Coordenadas Y visuales de cada estudiante
        cols = []     # Color del punto según el espacio donde está

        for agente in modelo.agents:
            if isinstance(agente, EstudianteAgent):
                xs.append(agente.px)                             # Posición X en el canvas
                ys.append(agente.py)                             # Posición Y en el canvas
                cols.append(COLORES_ESPACIO[agente.ubicacion])  # Color por espacio

        # Actualizamos el scatter con las nuevas posiciones y colores
        scatter.set_offsets(list(zip(xs, ys)))  # Nuevas coordenadas (x, y) de los puntos
        scatter.set_color(cols)                  # Nuevos colores de los puntos

        # --- Actualizamos el texto del paso actual ---
        texto_paso.set_text(f"Paso: {frame}  /  {num_pasos}")

        # --- Contamos estudiantes por espacio para los textos y barras ---
        n_aula = sum(1 for a in modelo.agents
                     if isinstance(a, EstudianteAgent) and a.ubicacion == "Aula")
        n_bib  = sum(1 for a in modelo.agents
                     if isinstance(a, EstudianteAgent) and a.ubicacion == "Biblioteca")
        n_caf  = sum(1 for a in modelo.agents
                     if isinstance(a, EstudianteAgent) and a.ubicacion == "Cafeteria")

        # Actualizamos el texto de conteo en la parte superior del campus
        texto_conteo.set_text(
            f"Aula: {n_aula}  |  Biblioteca: {n_bib}  |  Cafeteria: {n_caf}"
            f"   (Total: {n_aula + n_bib + n_caf})"
        )

        # --- Actualizamos las barras de ocupación ---
        valores = [n_aula, n_bib, n_caf]
        for i, (barra, valor) in enumerate(zip(barras, valores)):
            barra.set_height(valor)         # Nueva altura de la barra
            etiquetas_barras[i].set_text(str(valor))  # Nuevo número encima
            etiquetas_barras[i].set_y(valor + 0.3)    # Posición Y del número

        # Retornamos los artistas modificados para que matplotlib los redibuje
        return [scatter, texto_paso, texto_conteo] + list(barras) + etiquetas_barras

    # ---- Creamos y lanzamos la animación ----

    # FuncAnimation llama a actualizar(frame) para frames 0, 1, 2, ..., num_pasos
    anim = FuncAnimation(
        fig,
        actualizar,
        frames=num_pasos + 1,   # +1 para incluir el frame 0 (estado inicial)
        interval=intervalo_ms,  # Milisegundos entre frames
        repeat=True,            # Repite la animación al terminar
        blit=False              # blit=False para compatibilidad en Windows
    )

    # Mostramos la ventana con la animación
    plt.show()

    # Retornamos el modelo para acceder a los datos si se necesita el resumen
    return modelo


def imprimir_resumen(modelo, num_pasos):
    """
    Imprime estadísticas finales de la simulación en consola.
    Se muestra después de cerrar la ventana de animación.
    """
    # Obtenemos el DataFrame con datos recolectados por paso
    datos = modelo.datacollector.get_model_vars_dataframe()

    print("\n" + "="*55)
    print("   RESUMEN DE LA SIMULACION - MOVIMIENTO ESTUDIANTIL")
    print("="*55)
    print(f"  Total de estudiantes : {modelo.num_estudiantes}")
    print(f"  Pasos simulados      : {num_pasos}")
    print(f"  Umbral de ocupacion  : {int(UMBRAL_OCUPACION * 100)}%")
    print(f"  Prob. cambio volunt. : {int(PROB_CAMBIO_VOLUNTARIO * 100)}%")
    print()
    print("  Estadisticas por espacio:")
    print("  " + "-"*50)
    print(f"  {'Espacio':<12} {'Promedio':>10} {'Maximo':>10} {'Final':>10} {'Cap.':>8}")
    print("  " + "-"*50)
    for espacio in ESPACIOS.keys():
        promedio = datos[espacio].mean()    # Ocupación promedio durante toda la simulación
        maximo   = datos[espacio].max()     # Pico máximo de ocupación
        final    = datos[espacio].iloc[-1]  # Ocupación en el último paso
        capacidad = ESPACIOS[espacio]["capacidad"]
        print(f"  {espacio:<12} {promedio:>10.1f} {maximo:>10} {final:>10} {capacidad:>8}")
    print("="*55 + "\n")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    # Número de pasos de la simulación (frames de la animación, sin contar el 0)
    NUM_PASOS = 30

    # Número de estudiantes agentes
    NUM_ESTUDIANTES = 100

    print("\n" + "="*55)
    print("  SIMULACION DE MOVIMIENTO ESTUDIANTIL - ANIMACION")
    print("="*55)
    print(f"  Estudiantes : {NUM_ESTUDIANTES}")
    print(f"  Pasos       : {NUM_PASOS}")
    print(f"  Intervalo   : 700 ms por frame")
    print(f"  Umbral      : {int(UMBRAL_OCUPACION*100)}% de ocupacion")
    print("  Cierra la ventana para ver el resumen en consola.")
    print("="*55)

    # Lanzamos la animación: crea el modelo, simula y muestra los frames
    modelo_final = animar_simulacion(
        num_pasos=NUM_PASOS,
        num_estudiantes=NUM_ESTUDIANTES,
        intervalo_ms=700   # 700ms por frame: velocidad legible
    )

    # Resumen estadístico tras cerrar la ventana
    imprimir_resumen(modelo_final, NUM_PASOS)
