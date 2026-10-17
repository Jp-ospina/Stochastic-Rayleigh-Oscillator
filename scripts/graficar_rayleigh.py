import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "simulacion.csv"
FIG_DIR = PROJECT_ROOT / "resultados"
MAX_ANIMATION_FRAMES = 300
FRAME_INTERVAL_MS = 40


def read_simulation(path):
    steps = []
    times = []
    positions = []
    velocities = []
    amplitudes = []
    phases = []

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"paso", "t", "x", "vx", "amplitud", "fase"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} debe contener las columnas {sorted(required_columns)}."
            )

        for row in reader:
            steps.append(int(row["paso"]))
            times.append(float(row["t"]))
            positions.append(float(row["x"]))
            velocities.append(float(row["vx"]))
            amplitudes.append(float(row["amplitud"]))
            phases.append(float(row["fase"]))

    data = {
        "steps": np.asarray(steps, dtype=int),
        "times": np.asarray(times, dtype=float),
        "x": np.asarray(positions, dtype=float),
        "vx": np.asarray(velocities, dtype=float),
        "amplitude": np.asarray(amplitudes, dtype=float),
        "phase": np.asarray(phases, dtype=float),
    }
    if data["times"].size == 0:
        raise ValueError(f"{path} no contiene estados de simulación.")
    if not all(np.isfinite(values).all() for values in data.values()):
        raise ValueError(f"{path} contiene valores no finitos.")
    if any(values.size != data["times"].size for values in data.values()):
        raise ValueError(f"{path} tiene filas con datos incompletos.")
    if np.any(np.diff(data["steps"]) <= 0) or np.any(np.diff(data["times"]) <= 0):
        if data["times"].size > 1:
            raise ValueError("Los pasos y tiempos del CSV deben ser crecientes.")
    return data


def save_static_plots(data, output_dir):
    times = data["times"]
    phase_unwrapped = np.unwrap(data["phase"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(times, data["x"], color="#1769aa", linewidth=1)
    axes[0, 0].set(title="Posición", xlabel="Tiempo t", ylabel="x(t)")
    axes[0, 1].plot(times, data["vx"], color="#d1495b", linewidth=1)
    axes[0, 1].set(title="Velocidad", xlabel="Tiempo t", ylabel="v(t)")
    axes[1, 0].plot(times, data["amplitude"], color="#4c956c", linewidth=1)
    axes[1, 0].set(title="Amplitud instantánea", xlabel="Tiempo t", ylabel="A(t)")
    axes[1, 1].plot(times, phase_unwrapped, color="#7b2cbf", linewidth=1)
    axes[1, 1].set(
        title="Fase desenrollada (derivada de la fase guardada)",
        xlabel="Tiempo t",
        ylabel="Fase φ(t) [rad]",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.savefig(output_dir / "rayleigh_series_temporales.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.plot(data["x"], data["vx"], color="#3973ac", linewidth=0.8)
    ax.scatter(data["x"][0], data["vx"][0], color="#2a9d8f", label="Inicial", zorder=3)
    ax.scatter(data["x"][-1], data["vx"][-1], color="#e76f51", label="Final", zorder=3)
    ax.set(
        title="Retrato de fase del oscilador de Rayleigh",
        xlabel="Posición x",
        ylabel="Velocidad v",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "rayleigh_retrato_fase.png", dpi=160)
    plt.close(fig)


def save_animation(data, output_dir):
    if not writers.is_available("pillow"):
        raise RuntimeError("Se necesita Pillow para guardar la animación GIF.")

    times = data["times"]
    x = data["x"]
    vx = data["vx"]
    frame_indices = np.unique(
        np.linspace(
            0,
            len(times) - 1,
            min(len(times), MAX_ANIMATION_FRAMES),
            dtype=int,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    time_axis, phase_axis = axes

    time_axis.plot(times, x, color="#d9e2ec", linewidth=0.8)
    time_line, = time_axis.plot([], [], color="#1769aa", linewidth=1.2)
    time_marker, = time_axis.plot([], [], "o", color="#e76f51")
    time_axis.set(
        title="Evolución de la posición",
        xlabel="Tiempo t",
        ylabel="x(t)",
        xlim=(times[0], times[-1] if times[-1] > times[0] else times[0] + 1),
    )
    x_padding = max(float(np.ptp(x)) * 0.08, 0.1)
    time_axis.set_ylim(float(np.min(x) - x_padding), float(np.max(x) + x_padding))
    time_axis.grid(alpha=0.25)

    phase_axis.plot(x, vx, color="#d9e2ec", linewidth=0.8)
    phase_line, = phase_axis.plot([], [], color="#3973ac", linewidth=1.2)
    phase_marker, = phase_axis.plot([], [], "o", color="#e76f51")
    phase_axis.set(
        title="Trayectoria en el espacio de fase",
        xlabel="Posición x",
        ylabel="Velocidad v",
    )
    x_padding = max(float(np.ptp(x)) * 0.08, 0.1)
    vx_padding = max(float(np.ptp(vx)) * 0.08, 0.1)
    phase_axis.set_xlim(float(np.min(x) - x_padding), float(np.max(x) + x_padding))
    phase_axis.set_ylim(float(np.min(vx) - vx_padding), float(np.max(vx) + vx_padding))
    phase_axis.grid(alpha=0.25)
    time_label = fig.suptitle("")

    def update(frame_number):
        index = int(frame_indices[frame_number])
        time_line.set_data(times[: index + 1], x[: index + 1])
        time_marker.set_data([times[index]], [x[index]])
        phase_line.set_data(x[: index + 1], vx[: index + 1])
        phase_marker.set_data([x[index]], [vx[index]])
        time_label.set_text(f"t = {times[index]:.6g}")
        return time_line, time_marker, phase_line, phase_marker, time_label

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=FRAME_INTERVAL_MS,
        blit=False,
        repeat=False,
    )
    animation.save(
        output_dir / "rayleigh_animacion.gif",
        writer="pillow",
        fps=1000 // FRAME_INTERVAL_MS,
        dpi=100,
    )
    plt.close(fig)


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de simulación: {CSV_PATH}")
    if not FIG_DIR.is_dir():
        raise FileNotFoundError(f"La carpeta de salida debe existir: {FIG_DIR}")

    data = read_simulation(CSV_PATH)
    save_static_plots(data, FIG_DIR)
    save_animation(data, FIG_DIR)
    print(
        f"Se generaron las gráficas y la animación en {FIG_DIR} "
        f"usando {len(data['times'])} estados."
    )


if __name__ == "__main__":
    main()
