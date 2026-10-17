import csv
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES = (
    PROJECT_ROOT / "src" / "main.cpp",
    PROJECT_ROOT / "src" / "main_sin_modularizar.cpp",
)
HEADER = PROJECT_ROOT / "include" / "rayleigh.h"
LEGACY_CSV = PROJECT_ROOT / "data" / "simulacion.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "ensamble"

EPSILON = 0.2
DT = 0.01
STEPS = 10000
SIGMAS = (0.0, 0.2, 0.5, 1.0)
REPLICAS = 30
BASE_SEED = 20260317
CONFIGURATIONS = (
    ("x0_mas1_v0_0", 1.0, 0.0),
    ("x0_0_v0_mas1", 0.0, 1.0),
    ("x0_menos1_v0_0", -1.0, 0.0),
)
METHODS = ((1, "Euler-Maruyama"), (3, "Heun-estocastico"))
METHOD_LABELS = {method: label for method, label in METHODS}
MAX_ANIMATION_FRAMES = 300


def compile_solver(executable):
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-pedantic",
        f"-I{PROJECT_ROOT / 'include'}",
        *(str(source) for source in SOURCES),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def run_trajectory(executable, x0, v0, sigma, method, seed):
    input_data = (
        f"{x0}\n{v0}\n{STEPS}\n{EPSILON}\n{sigma}\n{DT}\n{method}\n{seed}\n"
    )
    result = subprocess.run(
        [str(executable)],
        input=input_data,
        text=True,
        capture_output=True,
        check=True,
        cwd=PROJECT_ROOT,
    )
    if not LEGACY_CSV.is_file():
        raise FileNotFoundError(f"El ejecutable no generó {LEGACY_CSV}.")

    with LEGACY_CSV.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        required = {"paso", "t", "x", "vx", "amplitud", "fase"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("El CSV de Rayleigh no contiene todas las columnas esperadas.")
        rows = list(reader)

    if len(rows) != STEPS + 1:
        raise ValueError(
            f"Se esperaban {STEPS + 1} estados y se leyeron {len(rows)}."
        )
    values = np.asarray(
        [
            [
                float(row["t"]),
                float(row["x"]),
                float(row["vx"]),
                float(row["amplitud"]),
                float(row["fase"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError(
            "La trayectoria contiene estados no finitos. "
            f"Salida del ejecutable:\n{result.stdout}\n{result.stderr}"
        )
    return values


def make_directories():
    directories = {}
    for configuration, _, _ in CONFIGURATIONS:
        root = OUTPUT_ROOT / configuration
        directories[configuration] = {
            "root": root,
            "csv": root / "CSV",
            "figures": root / "Fig",
            "animations": root / "Animaciones",
        }
        for directory in directories[configuration].values():
            directory.mkdir(parents=True, exist_ok=True)
    return directories


def write_metadata():
    path = OUTPUT_ROOT / "parametros_ensamble.csv"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["parametro", "valor"])
        writer.writerows(
            [
                ("epsilon", EPSILON),
                ("dt", DT),
                ("pasos", STEPS),
                ("tiempo_final", STEPS * DT),
                ("sigmas", ";".join(str(value) for value in SIGMAS)),
                ("replicas_por_sigma_y_metodo", REPLICAS),
                ("semilla_base", BASE_SEED),
                ("metodos", ";".join(label for _, label in METHODS)),
                ("tiempos_distribucion_x", "2.5;5;7.5;10"),
                ("condiciones_iniciales", "x0_mas1_v0_0; x0_0_v0_mas1; x0_menos1_v0_0"),
                *[
                    (
                        f"condicion_{name}",
                        f"x0={x0};vx0={v0}",
                    )
                    for name, x0, v0 in CONFIGURATIONS
                ],
            ]
        )
    return path


def create_accumulator():
    shape = (STEPS + 1,)
    return {
        "sum_x": np.zeros(shape),
        "sum_x2": np.zeros(shape),
        "sum_vx": np.zeros(shape),
        "sum_vx2": np.zeros(shape),
        "sum_amplitude": np.zeros(shape),
        "sum_sin_phase": np.zeros(shape),
        "sum_cos_phase": np.zeros(shape),
        "times": None,
        "final_x": [],
        "final_vx": [],
        "sample": None,
        "count": 0,
    }


def write_trajectory(writer, configuration, sigma, method, replica, seed, values):
    label = METHOD_LABELS[method]
    for step, row in enumerate(values):
        time, position, velocity, amplitude, phase = row
        writer.writerow(
            [
                configuration,
                sigma,
                label,
                replica,
                seed,
                step,
                time,
                position,
                velocity,
                amplitude,
                phase,
            ]
        )


def write_statistics(path, accumulators):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "sigma",
                "t",
                "media_x",
                "varianza_x",
                "media_vx",
                "varianza_vx",
                "media_amplitud",
                "fase_circular_media",
                "replicas",
            ]
        )
        for sigma in SIGMAS:
            accumulator = accumulators[sigma]
            count = accumulator["count"]
            mean_x = accumulator["sum_x"] / count
            variance_x = np.maximum(
                accumulator["sum_x2"] / count - mean_x * mean_x, 0.0
            )
            mean_vx = accumulator["sum_vx"] / count
            variance_vx = np.maximum(
                accumulator["sum_vx2"] / count - mean_vx * mean_vx, 0.0
            )
            mean_amplitude = accumulator["sum_amplitude"] / count
            mean_phase = np.arctan2(
                accumulator["sum_sin_phase"],
                accumulator["sum_cos_phase"],
            )
            for index, time in enumerate(accumulator["times"]):
                writer.writerow(
                    [
                        sigma,
                        time,
                        mean_x[index],
                        variance_x[index],
                        mean_vx[index],
                        variance_vx[index],
                        mean_amplitude[index],
                        mean_phase[index],
                        count,
                    ]
                )


def plot_statistics(configuration, method, accumulators, directories):
    label = METHOD_LABELS[method]
    figure_dir = directories[configuration]["figures"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for sigma in SIGMAS:
        accumulator = accumulators[sigma]
        count = accumulator["count"]
        mean_x = accumulator["sum_x"] / count
        variance_x = np.maximum(
            accumulator["sum_x2"] / count - mean_x * mean_x, 0.0
        )
        axes[0].plot(accumulator["times"], mean_x, label=f"σ={sigma:g}")
        axes[1].plot(accumulator["times"], variance_x, label=f"σ={sigma:g}")
    axes[0].set_ylabel("⟨x(t)⟩")
    axes[0].set_title(f"Media del ensamble — {label}")
    axes[1].set_ylabel("Var[x(t)]")
    axes[1].set_xlabel("Tiempo t")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
    fig.savefig(figure_dir / f"estadisticas_{label}.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for sigma in SIGMAS:
        accumulator = accumulators[sigma]
        axes[0].hist(
            accumulator["final_x"],
            bins=18,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=f"σ={sigma:g}",
        )
        axes[1].scatter(
            accumulator["final_x"],
            accumulator["final_vx"],
            s=13,
            alpha=0.65,
            label=f"σ={sigma:g}",
        )
    axes[0].set(title="Distribución final de x", xlabel="x(T)", ylabel="Densidad")
    axes[1].set(title="Ensamble final en espacio de fase", xlabel="x(T)", ylabel="v(T)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(figure_dir / f"distribucion_final_{label}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for sigma in SIGMAS:
        sample = accumulators[sigma]["sample"]
        ax.plot(sample[:, 1], sample[:, 2], linewidth=0.9, label=f"σ={sigma:g}")
    ax.set(
        title=f"Trayectorias representativas — {label}",
        xlabel="x",
        ylabel="v",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(figure_dir / f"retrato_fase_{label}.png", dpi=160)
    plt.close(fig)


def plot_distributions_over_time(configuration, method, directories):
    label = METHOD_LABELS[method]
    csv_path = (
        directories[configuration]["csv"]
        / f"trayectorias_{label}.csv"
    )
    snapshot_steps = (250, 500, 750, 1000)
    samples = {
        (sigma, step): []
        for sigma in SIGMAS
        for step in snapshot_steps
    }
    with csv_path.open("r", newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            step = int(row["paso"])
            key = (float(row["sigma"]), step)
            if key in samples:
                samples[key].append(float(row["x"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, step in zip(axes.flat, snapshot_steps):
        all_positions = [
            position
            for sigma in SIGMAS
            for position in samples[(sigma, step)]
        ]
        bins = np.histogram_bin_edges(all_positions, bins=16)
        for sigma in SIGMAS:
            axis.hist(
                samples[(sigma, step)],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=f"σ={sigma:g}",
            )
        axis.set(
            title=f"P(x,t={step * DT:g})",
            xlabel="x",
            ylabel="Densidad de probabilidad",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(f"Distribución de posición en varios tiempos — {label}")
    fig.savefig(
        directories[configuration]["figures"]
        / f"distribucion_x_en_el_tiempo_{label}.png",
        dpi=160,
    )
    plt.close(fig)


def save_animation(method, accumulators, output_path):
    if not writers.is_available("pillow"):
        raise RuntimeError("Se necesita Pillow para guardar las animaciones GIF.")

    times = accumulators[SIGMAS[0]]["times"]
    frame_indices = np.unique(
        np.linspace(
            0,
            STEPS,
            min(STEPS + 1, MAX_ANIMATION_FRAMES),
            dtype=int,
        )
    )
    means = {
        sigma: accumulators[sigma]["sum_x"] / accumulators[sigma]["count"]
        for sigma in SIGMAS
    }
    mean_velocities = {
        sigma: accumulators[sigma]["sum_vx"] / accumulators[sigma]["count"]
        for sigma in SIGMAS
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    position_axis, phase_axis = axes
    lines = []
    phase_lines = []
    for sigma in SIGMAS:
        line, = position_axis.plot([], [], label=f"σ={sigma:g}")
        phase_line, = phase_axis.plot([], [], label=f"σ={sigma:g}")
        lines.append(line)
        phase_lines.append(phase_line)
    position_axis.set(
        title=f"Media ⟨x(t)⟩ — {METHOD_LABELS[method]}",
        xlabel="Tiempo t",
        ylabel="⟨x(t)⟩",
        xlim=(times[0], times[-1]),
    )
    position_axis.set_ylim(
        min(float(np.min(values)) for values in means.values()) - 0.1,
        max(float(np.max(values)) for values in means.values()) + 0.1,
    )
    phase_x = np.concatenate(list(means.values()))
    phase_y = np.concatenate(list(mean_velocities.values()))
    phase_axis.set(
        title="Trayectoria de las medias del ensamble",
        xlabel="⟨x(t)⟩",
        ylabel="⟨v(t)⟩",
        xlim=(float(phase_x.min() - 0.1), float(phase_x.max() + 0.1)),
        ylim=(float(phase_y.min() - 0.1), float(phase_y.max() + 0.1)),
    )
    phase_axis.grid(alpha=0.25)
    phase_axis.legend()
    position_axis.grid(alpha=0.25)
    position_axis.legend()
    title = fig.suptitle("")

    def update(frame_number):
        index = int(frame_indices[frame_number])
        for line, phase_line, sigma in zip(lines, phase_lines, SIGMAS):
            mean_x = means[sigma]
            mean_vx = mean_velocities[sigma]
            line.set_data(times[: index + 1], mean_x[: index + 1])
            phase_line.set_data(mean_x[: index + 1], mean_vx[: index + 1])
        title.set_text(f"t = {times[index]:.3f}")
        return (*lines, *phase_lines, title)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=50,
        blit=False,
        repeat=False,
    )
    animation.save(output_path, writer="pillow", fps=20, dpi=90)
    plt.close(fig)


def analyze_configuration(executable, configuration, x0, v0, directories):
    csv_dir = directories[configuration]["csv"]
    output_files = {
        method: (csv_dir / f"trayectorias_{METHOD_LABELS[method]}.csv").open(
            "w", newline="", encoding="utf-8"
        )
        for method, _ in METHODS
    }
    writers_by_method = {
        method: csv.writer(output_file)
        for method, output_file in output_files.items()
    }
    for writer in writers_by_method.values():
        writer.writerow(
            [
                "configuracion",
                "sigma",
                "metodo",
                "replica",
                "semilla",
                "paso",
                "t",
                "x",
                "vx",
                "amplitud",
                "fase",
            ]
        )
    accumulators = {
        method: {sigma: create_accumulator() for sigma in SIGMAS}
        for method, _ in METHODS
    }

    try:
        for method, _ in METHODS:
            for sigma_index, sigma in enumerate(SIGMAS):
                accumulator = accumulators[method][sigma]
                for replica in range(REPLICAS):
                    seed = (
                        BASE_SEED
                        + CONFIGURATIONS.index((configuration, x0, v0)) * 100000
                        + sigma_index * 1000
                        + replica
                    )
                    values = run_trajectory(
                        executable, x0, v0, sigma, method, seed
                    )
                    if accumulator["times"] is None:
                        accumulator["times"] = values[:, 0].copy()
                    elif not np.array_equal(accumulator["times"], values[:, 0]):
                        raise ValueError("Los tiempos no coinciden entre réplicas.")

                    accumulator["sum_x"] += values[:, 1]
                    accumulator["sum_x2"] += values[:, 1] ** 2
                    accumulator["sum_vx"] += values[:, 2]
                    accumulator["sum_vx2"] += values[:, 2] ** 2
                    accumulator["sum_amplitude"] += values[:, 3]
                    accumulator["sum_sin_phase"] += np.sin(values[:, 4])
                    accumulator["sum_cos_phase"] += np.cos(values[:, 4])
                    accumulator["final_x"].append(values[-1, 1])
                    accumulator["final_vx"].append(values[-1, 2])
                    if replica == 0:
                        accumulator["sample"] = values.copy()
                    accumulator["count"] += 1
                    write_trajectory(
                        writers_by_method[method],
                        configuration,
                        sigma,
                        method,
                        replica,
                        seed,
                        values,
                    )
    finally:
        for output_file in output_files.values():
            output_file.close()

    for method, _ in METHODS:
        statistics_path = (
            csv_dir / f"estadisticas_{METHOD_LABELS[method]}.csv"
        )
        write_statistics(statistics_path, accumulators[method])
        plot_statistics(configuration, method, accumulators[method], directories)
        plot_distributions_over_time(configuration, method, directories)
        animation_path = (
            directories[configuration]["animations"]
            / f"animacion_ensamble_{METHOD_LABELS[method]}.gif"
        )
        save_animation(method, accumulators[method], animation_path)


def main():
    for source in (*SOURCES, HEADER):
        if not source.is_file():
            raise FileNotFoundError(f"No se encontró el archivo requerido: {source}")
    if not LEGACY_CSV.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de salida: {LEGACY_CSV}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    directories = make_directories()
    metadata_path = write_metadata()
    original_csv = LEGACY_CSV.read_bytes()
    try:
        with tempfile.TemporaryDirectory(prefix="rayleigh_") as temporary_directory:
            executable = Path(temporary_directory) / "rayleigh.exe"
            compile_solver(executable)
            for configuration, x0, v0 in CONFIGURATIONS:
                analyze_configuration(
                    executable, configuration, x0, v0, directories
                )
    finally:
        LEGACY_CSV.write_bytes(original_csv)

    print(
        f"Ensambles generados en {OUTPUT_ROOT}: "
        f"{len(CONFIGURATIONS)} condiciones iniciales, {len(SIGMAS)} valores "
        f"de sigma, {REPLICAS} replicas y {len(METHODS)} metodos."
    )
    print(f"Parametros registrados en {metadata_path}.")


if __name__ == "__main__":
    main()
