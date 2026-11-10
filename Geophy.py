import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
from scipy.interpolate import griddata

import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTADO DE SESIÓN
# ==============================================================================
st.set_page_config(page_title="Suite Hidrogeofísica - Módulo 1: TRE", layout="wide")

if "modelos_geofisicos" not in st.session_state:
    st.session_state["modelos_geofisicos"] = {}

if "modelos_previos" not in st.session_state:
    # Guarda la inversión inmediata anterior para comparar antes/después
    st.session_state["modelos_previos"] = {}

if "raw_ert_datasets" not in st.session_state:
    st.session_state["raw_ert_datasets"] = {}

if "crs_datasets" not in st.session_state:
    st.session_state["crs_datasets"] = {}

if "residual_masks" not in st.session_state:
    # Almacena máscaras de poda de residuales post-inversión (estilo EarthImager)
    st.session_state["residual_masks"] = {}


# ==============================================================================
# UTILIDADES PARA ESCALAS DE COLOR, AUTO-LIMPIEZA Y CRITERIO GEOFÍSICO EXPERTO
# ==============================================================================
def generar_ticks_ohm_m(r_min, r_max, n_ticks=8):
    """
    Genera posiciones en log10 pero con etiquetas legibles en notación normal (Ω·m)
    estilo EarthImager / RES2DINV (ej. 1, 2, 5, 10, 20, 50, 100, 200, 500).
    """
    r_min = max(0.1, float(r_min))
    r_max = max(r_min * 1.5, float(r_max))

    base_steps = [1, 1.5, 2, 3, 5, 7.5]
    candidates = []
    for exp in range(-1, 5):
        for b in base_steps:
            val = b * (10 ** exp)
            if r_min <= val <= r_max:
                candidates.append(val)

    if len(candidates) < 4:
        vals = np.geomspace(r_min, r_max, n_ticks)
    else:
        vals = [r_min] + candidates + [r_max]
        vals = np.unique(np.round(vals, 1))

    tick_vals_log = np.log10(vals)
    tick_text = [f"{v:g} Ω·m" if v >= 10 else f"{v:.1f} Ω·m" for v in vals]
    return tick_vals_log, tick_text


def calcular_autolimpieza_iqr(rhoa_array, factor_iqr=1.5):
    """
    Calcula umbrales óptimos de limpieza usando el Rango Intercuartílico (IQR)
    sobre log10(rhoa) para eliminar picos aberrantes conservando la estructura geológica.
    """
    rhoa_pos = rhoa_array[rhoa_array > 0]
    log_r = np.log10(rhoa_pos)

    q25, q75 = np.percentile(log_r, [25, 75])
    iqr = q75 - q25

    low_log = q25 - (factor_iqr * iqr)
    high_log = q75 + (factor_iqr * iqr)

    r_min_sug = max(float(np.min(rhoa_pos)), float(10 ** low_log))
    r_max_sug = min(float(np.max(rhoa_pos)), float(10 ** high_log))

    r_min_sug = float(np.floor(r_min_sug * 100.0) / 100.0)
    r_max_sug = float(np.ceil(r_max_sug * 100.0) / 100.0)

    mask_out = (rhoa_array < r_min_sug) | (rhoa_array > r_max_sug)
    n_removidos = int(np.sum(mask_out))

    return r_min_sug, r_max_sug, n_removidos


def diagnosticar_parametros_geofisicos(data_ert, elec_d):
    """
    Analiza la geometría del tendido, el factor geométrico K y la estadística de
    resistividad aparente para recomendar los parámetros óptimos de inversión
    que elegiría un especialista en geofísica.
    """
    n_elecs = data_ert.sensorCount()
    L = float(elec_d.max() - elec_d.min())
    diffs = np.diff(np.sort(np.unique(elec_d)))
    a_min = float(np.median(diffs)) if len(diffs) > 0 else 5.0

    rhoa = np.array(data_ert["rhoa"])
    rhoa_pos = rhoa[rhoa > 0]
    log_rho = np.log10(rhoa_pos)

    std_log = float(np.std(log_rho))
    q10, q25, q50, q75, q90 = np.percentile(log_rho, [10, 25, 50, 75, 90])
    iqr_log = max(0.05, q75 - q25)
    tail_ratio = (q90 - q10) / iqr_log  # Mide peso de colas / ruido impulsivo

    # 1. Estimación de Profundidad de Investigación (DOI efectivo según geometría)
    # En Dipolo-Dipolo / Schlumberger multielectródico de 56+ electrodos alcanza ~0.24 - 0.25 L
    factor_prof = 0.245 if n_elecs >= 48 else 0.21
    prof_sugerida = float(np.round((L * factor_prof) / 2.0) * 2.0)
    prof_sugerida = max(20.0, prof_sugerida)

    # 2. Selección de Esquema de Inversión (L1 vs L2) y Regularización (Lambda)
    if tail_ratio > 2.1 or std_log > 0.45:
        esquema_idx = 1  # Robusta en Datos L1 (Ignora outliers)
        esquema_txt = "Robusta en Datos L1 (Ignora outliers)"
        lam_sug = 15.0 if std_log > 0.55 else 20.0
        motivo_esquema = (
            f"Alta dispersión en lecturas (σ_log={std_log:.2f}, colas={tail_ratio:.2f}). "
            "Se recomienda Norma L1 en datos para evitar artefactos por picos de voltaje en niveles profundos."
        )
    elif std_log > 0.35 and tail_ratio <= 1.8:
        esquema_idx = 2  # Bloques L1
        esquema_txt = "Bloques / Contactos Abruptos (Blocky L1)"
        lam_sug = 15.0
        motivo_esquema = (
            "Contrastes marcados con bajo ruido aleatorio. "
            "El esquema de Bloques L1 resalta contactos litológicos o fallas."
        )
    else:
        esquema_idx = 0  # Suavizada L2 (Occam)
        esquema_txt = "Suavizada L2 (Occam / Smooth)"
        lam_sug = 25.0
        motivo_esquema = (
            f"Distribución suave y estable (σ_log={std_log:.2f}). "
            "El esquema Occam L2 es ideal para transiciones graduales."
        )

    # 3. Anisotropía Estructural (zWeight) según ambiente hidrogeológico inferido
    mediana_ohm = float(10 ** q50)
    if mediana_ohm < 80.0:
        z_weight_sug = 0.25
        ambiente_txt = "Acuífero sedimentario / aluvial / costero (conductivo)"
    elif mediana_ohm < 250.0:
        z_weight_sug = 0.35
        ambiente_txt = "Medio transicional / vulcanosedimentario"
    else:
        z_weight_sug = 0.50
        ambiente_txt = "Medio resistivo / roca fracturada"

    # 4. Refinamiento de malla (paraDX) e iteraciones
    para_dx_sug = 0.33 if n_elecs <= 64 else 0.5
    iter_sug = 12 if esquema_idx == 1 else 10

    return {
        "L": L,
        "a_min": a_min,
        "prof_sug": prof_sugerida,
        "lam_sug": lam_sug,
        "z_weight_sug": z_weight_sug,
        "esquema_idx": esquema_idx,
        "esquema_txt": esquema_txt,
        "para_dx_sug": para_dx_sug,
        "iter_sug": iter_sug,
        "cov_sug": -2.5,
        "motivo_esquema": motivo_esquema,
        "ambiente_txt": ambiente_txt,
        "mediana_ohm": mediana_ohm,
    }


# ==============================================================================
# PARSERS ESPECIALIZADOS (AGI SUPERSTING .STG / .CRS Y MULTIFORMATO)
# ==============================================================================
def parsear_supersting_stg_manual(filepath):
    """Parser directo para formato ASCII de AGI SuperSting (.stg)."""
    registros = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 21 and parts[0].isdigit():
                try:
                    v_over_i = float(parts[4])
                    err_permil = float(parts[5])
                    curr_ma = float(parts[6])
                    rhoa = float(parts[7])
                    ax, ay, az = float(parts[9]), float(parts[10]), float(parts[11])
                    bx, by, bz = float(parts[12]), float(parts[13]), float(parts[14])
                    mx, my, mz = float(parts[15]), float(parts[16]), float(parts[17])
                    nx, ny, nz = float(parts[18]), float(parts[19]), float(parts[20])
                    registros.append([
                        ax, ay, az, bx, by, bz, mx, my, mz, nx, ny, nz,
                        v_over_i, rhoa, err_permil / 1000.0, curr_ma
                    ])
                except ValueError:
                    continue

    if not registros:
        raise ValueError("No se encontraron registros válidos en el archivo .stg.")

    arr = np.array(registros)
    all_pts = np.vstack([arr[:, 0:3], arr[:, 3:6], arr[:, 6:9], arr[:, 9:12]])
    unique_elecs = np.unique(np.round(all_pts, 3), axis=0)
    order = np.lexsort((unique_elecs[:, 2], unique_elecs[:, 1], unique_elecs[:, 0]))
    unique_elecs = unique_elecs[order]

    data = pg.DataContainerERT()
    for e in unique_elecs:
        data.createSensor(pg.Pos(e[0], e[1], e[2]))

    def find_idx(pts):
        dists = np.linalg.norm(unique_elecs[None, :, :] - pts[:, None, :], axis=2)
        return np.argmin(dists, axis=1)

    data.resize(len(arr))
    data["a"] = find_idx(arr[:, 0:3])
    data["b"] = find_idx(arr[:, 3:6])
    data["m"] = find_idx(arr[:, 6:9])
    data["n"] = find_idx(arr[:, 9:12])
    data["r"] = arr[:, 12]
    data["rhoa"] = np.abs(arr[:, 13])
    data["err"] = np.clip(arr[:, 14], 0.005, 0.5)
    data["i"] = arr[:, 15] / 1000.0
    data["valid"] = 1
    return data


def parsear_supersting_crs(uploaded_file):
    """Extrae los valores de resistencia de contacto desde un archivo .crs."""
    filas = []
    content = uploaded_file.getvalue().decode("latin-1", errors="ignore").splitlines()
    for line in content:
        parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        if len(nums) >= 2:
            filas.append(nums)

    if not filas:
        return pd.DataFrame()

    df_crs = pd.DataFrame(filas).iloc[:, :4]
    cols = [f"Col_{i+1}" for i in range(df_crs.shape[1])]
    if df_crs.shape[1] >= 2:
        cols[0] = "Electrodo"
        cols[-1] = "Resistencia_Contacto_Ohm"
    df_crs.columns = cols
    return df_crs


def cargar_archivo_tre(uploaded_file):
    """Lee archivos .stg, .dat, .ohm, .txt usando PyGIMLi con fallback manual."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        try:
            data = ert.load(tmp_path)
        except Exception:
            if suffix == ".stg":
                data = parsear_supersting_stg_manual(tmp_path)
            else:
                raise

        if not data.haveData("k") or np.all(np.array(data["k"]) == 0):
            data["k"] = ert.createGeometricFactors(data, numerical=True)
        if not data.haveData("rhoa") or np.all(np.array(data["rhoa"]) == 0):
            if data.haveData("r"):
                data["rhoa"] = np.abs(data["r"] * data["k"])
            elif data.haveData("u") and data.haveData("i"):
                data["rhoa"] = np.abs((data["u"] / data["i"]) * data["k"])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return data


def generar_tre_sintetica():
    """Genera un dataset sintético Dipolo-Dipolo con topografía para pruebas."""
    x_elec = np.linspace(0, 200, 41)
    z_elec = 1520.0 + 8.0 * np.sin(x_elec / 60.0) - 0.02 * x_elec
    pos = np.column_stack([x_elec, z_elec])

    scheme = ert.createData(elecs=pos, schemeName="dd")
    world = mt.createWorld(start=[-50, 1440], end=[250, 1530], worldMarker=True)
    bloque = mt.createPolygon(
        [[60, 1485], [140, 1485], [140, 1505], [60, 1505]],
        isClosed=True, marker=2
    )
    geom = world + bloque
    mesh_sim = mt.createMesh(geom, quality=33, area=5.0)

    rhomap = [[1, 45.0], [2, 12.0]]
    data_sim = ert.simulate(
        mesh_sim, scheme=scheme, res=rhomap, noiseLevel=0.02, noiseAbs=1e-5, seed=42
    )
    data_sim["rhoa"] = np.abs(data_sim["rhoa"])
    return data_sim


# ==============================================================================
# MOTOR DE INVERSIÓN Y GEORREFERENCIACIÓN (Cacheado en RAM)
# ==============================================================================
@st.cache_data(show_spinner=False)
def ejecutar_inversion_tre(
    _data_filtered, section_id, n_valid, lam, z_weight, robust_data, blocky_model, max_iter, para_dx, max_depth
):
    """
    Ejecuta la inversión 2D en PyGIMLi y devuelve resistividad por celda, por vértice,
    respuesta calculada (rhoa_calc) y residuales individuales para optimizar el RMS.
    """
    mgr = ert.ERTManager(_data_filtered, verbose=False)

    inv_kwargs = {
        "lam": lam,
        "zWeight": z_weight,
        "robustData": robust_data,
        "blockyModel": blocky_model,
        "maxIter": max_iter,
        "paraDX": para_dx,
        "verbose": False,
    }
    if max_depth > 0:
        inv_kwargs["paraDepth"] = max_depth

    model_res = mgr.invert(**inv_kwargs)
    mesh_2d = mgr.paraDomain
    coverage = mgr.coverage()

    nodes_2d = np.array([[n.x(), n.y()] for n in mesh_2d.nodes()])
    res_arr = np.array(model_res)
    cov_arr = np.array(coverage)

    node_res = np.array(mt.cellDataToNodeData(mesh_2d, res_arr))
    node_cov = np.array(mt.cellDataToNodeData(mesh_2d, cov_arr))

    triangles = []
    cell_res = []
    cell_cov = []

    for idx, cell in enumerate(mesh_2d.cells()):
        n_ids = [cell.node(i).id() for i in range(cell.nodeCount())]
        if len(n_ids) == 3:
            triangles.append(n_ids)
            cell_res.append(res_arr[idx])
            cell_cov.append(cov_arr[idx])
        elif len(n_ids) == 4:
            triangles.append([n_ids[0], n_ids[1], n_ids[2]])
            triangles.append([n_ids[0], n_ids[2], n_ids[3]])
            cell_res.extend([res_arr[idx], res_arr[idx]])
            cell_cov.extend([cov_arr[idx], cov_arr[idx]])

    centers_2d = np.array([[c.x(), c.y()] for c in mesh_2d.cellCenters()])

    # Respuesta directa del modelo invertido y análisis de residuales por cuadrupolo
    rhoa_obs = np.array(_data_filtered["rhoa"])
    rhoa_calc = np.array(mgr.inv.response)
    res_pct = 100.0 * (rhoa_calc - rhoa_obs) / np.maximum(1e-4, rhoa_obs)

    corr_matrix = np.corrcoef(np.log10(rhoa_obs), np.log10(np.maximum(1e-4, rhoa_calc)))
    r2_log = float(corr_matrix[0, 1] ** 2) if corr_matrix.shape == (2, 2) else 0.0

    metrics = {
        "rms_rel": float(mgr.inv.relrms()),
        "chi2": float(mgr.inv.chi2()),
        "r2_log": r2_log,
        "n_cells": mesh_2d.cellCount(),
        "n_data": int(n_valid),
        "lam": float(lam),
        "z_weight": float(z_weight),
    }

    return (
        nodes_2d,
        np.array(triangles),
        np.array(cell_res),
        np.array(cell_cov),
        node_res,
        node_cov,
        centers_2d,
        res_arr,
        cov_arr,
        rhoa_obs,
        rhoa_calc,
        res_pct,
        metrics,
    )


def interpolar_malla_suave_2d(
    centers_2d, res_arr, cov_arr, elec_d, elec_z, z_offset, cov_threshold, usar_recorte_doi=False, nx=240, nz=120
):
    """
    Interpola los centroides de elementos finitos hacia una malla 2D regular suave
    mostrando el trapecio completo hasta la profundidad máxima de la malla.
    """
    d_pts = centers_2d[:, 0]
    z_pts = centers_2d[:, 1] + z_offset

    d_min, d_max = elec_d.min(), elec_d.max()
    d_grid = np.linspace(d_min, d_max, nx)

    z_min_mesh = z_pts.min()
    z_max_mesh = (elec_z + z_offset).max()
    z_grid = np.linspace(z_min_mesh, z_max_mesh, nz)
    D, Z = np.meshgrid(d_grid, z_grid)

    log_rho_lin = griddata((d_pts, z_pts), np.log10(res_arr), (D, Z), method="linear")
    log_rho_near = griddata((d_pts, z_pts), np.log10(res_arr), (D, Z), method="nearest")
    log_rho_grid = np.where(np.isnan(log_rho_lin), log_rho_near, log_rho_lin)

    cov_grid = griddata((d_pts, z_pts), cov_arr, (D, Z), method="nearest")

    z_topo_interp = np.interp(d_grid, elec_d, elec_z + z_offset)
    Prof_bajo_terreno = z_topo_interp[None, :] - Z

    dist_borde_izq = D - d_min
    dist_borde_der = d_max - D
    mask_trapezoid = (Prof_bajo_terreno > dist_borde_izq * 1.15) | (Prof_bajo_terreno > dist_borde_der * 1.15)

    mask_invalid = (Z > z_topo_interp[None, :]) | mask_trapezoid
    if usar_recorte_doi:
        mask_invalid = mask_invalid | (cov_grid < cov_threshold)

    log_rho_grid[mask_invalid] = np.nan
    rho_grid = 10 ** log_rho_grid

    return d_grid, z_grid, rho_grid, log_rho_grid


def proyectar_2d_a_utm(coords_2d, x_ini, y_ini, x_fin, y_fin, z_offset=0.0):
    """Transforma coordenadas locales (d, z) a coordenadas reales (X_UTM, Y_UTM, Z_msnm)."""
    d_local = coords_2d[:, 0]
    z_local = coords_2d[:, 1] + z_offset

    d_min, d_max = d_local.min(), d_local.max()
    L = d_max - d_min if (d_max - d_min) > 0 else 1.0
    t = (d_local - d_min) / L

    x_utm = x_ini + t * (x_fin - x_ini)
    y_utm = y_ini + t * (y_fin - y_ini)
    return np.column_stack([x_utm, y_utm, z_local])


def crear_figura_plotly_2d(
    d_grid, y_grid_plot, rho_grid, log_rho_grid, elec_d, y_elecs_plot,
    r_min_user, r_max_user, modo_escala, paleta_color, mostrar_isolineas,
    y_label, exag_2d, titulo
):
    """Construye la figura interactiva Plotly de una sección 2D."""
    fig = go.Figure()

    if modo_escala.startswith("Proporcional Logarítmica"):
        tick_vals_log, tick_text_ohm = generar_ticks_ohm_m(r_min_user, r_max_user)
        fig.add_trace(
            go.Contour(
                x=d_grid,
                y=y_grid_plot,
                z=log_rho_grid,
                customdata=rho_grid,
                zmin=np.log10(r_min_user),
                zmax=np.log10(r_max_user),
                colorscale=paleta_color,
                contours=dict(
                    coloring="heatmap",
                    showlines=mostrar_isolineas,
                    start=np.log10(r_min_user),
                    end=np.log10(r_max_user),
                    size=(np.log10(r_max_user) - np.log10(r_min_user)) / 14.0,
                ),
                line=dict(width=0.6, color="rgba(0,0,0,0.35)"),
                colorbar=dict(
                    title="Resistividad (Ω·m)",
                    tickvals=tick_vals_log,
                    ticktext=tick_text_ohm,
                ),
                hovertemplate=(
                    "<b>Distancia:</b> %{x:.1f} m<br>"
                    f"<b>{y_label}:</b> %{{y:.1f}} m<br>"
                    "<b>Resistividad:</b> %{customdata:.1f} Ω·m<extra></extra>"
                ),
                name="TRE Suavizada",
            )
        )
    else:
        fig.add_trace(
            go.Contour(
                x=d_grid,
                y=y_grid_plot,
                z=rho_grid,
                zmin=r_min_user,
                zmax=r_max_user,
                colorscale=paleta_color,
                contours=dict(
                    coloring="heatmap",
                    showlines=mostrar_isolineas,
                    start=r_min_user,
                    end=r_max_user,
                    size=(r_max_user - r_min_user) / 14.0,
                ),
                line=dict(width=0.6, color="rgba(0,0,0,0.35)"),
                colorbar=dict(title="Resistividad (Ω·m)", ticksuffix=" Ω·m"),
                hovertemplate=(
                    "<b>Distancia:</b> %{x:.1f} m<br>"
                    f"<b>{y_label}:</b> %{{y:.1f}} m<br>"
                    "<b>Resistividad:</b> %{z:.1f} Ω·m<extra></extra>"
                ),
                name="TRE Lineal",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=elec_d,
            y=y_elecs_plot,
            mode="markers+lines",
            marker=dict(symbol="triangle-down", size=6, color="black"),
            line=dict(color="black", width=1.5),
            name="Electrodos",
        )
    )

    valid_rows = ~np.all(np.isnan(rho_grid), axis=1)
    y_min_valid = float(np.nanmin(y_grid_plot[valid_rows])) if np.any(valid_rows) else float(np.min(y_grid_plot))
    y_max_valid = float(np.nanmax(y_elecs_plot)) + 3.0

    fig.update_layout(
        title=titulo,
        xaxis_title="Distancia sobre la línea (m)",
        yaxis_title=y_label,
        height=500,
        xaxis=dict(range=[elec_d.min() - 2, elec_d.max() + 2]),
        yaxis=dict(
            range=[y_min_valid - 2, y_max_valid],
            scaleanchor="x",
            scaleratio=exag_2d,
        ),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig


# ==============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ==============================================================================
st.title("⚡ Módulo 1: Procesamiento e Inversión 2D/3D de TRE")
st.markdown(
    "Carga archivos crudos de resistividad eléctrica (**AGI SuperSting .stg / .crs**, "
    "**RES2DINV .dat**, **.ohm**), aplica auto-limpieza, utiliza el **Asistente Geofísico de Inversión** "
    "para optimizar el RMS y visualiza secciones suavizadas en notación normal (**Ω·m**)."
)

with st.sidebar:
    st.header("1. Ingesta de Secciones TRE")
    uploaded_files = st.file_uploader(
        "Subir archivos (.stg, .crs, .dat, .ohm, .txt)",
        type=["stg", "crs", "dat", "ohm", "txt", "udf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for u_file in uploaded_files:
            base_name, ext = os.path.splitext(u_file.name)
            ext = ext.lower()
            if ext == ".crs":
                if base_name not in st.session_state["crs_datasets"]:
                    st.session_state["crs_datasets"][base_name] = parsear_supersting_crs(u_file)
                    st.toast(f"Resistencia de contacto cargada: {u_file.name}")
            else:
                if base_name not in st.session_state["raw_ert_datasets"]:
                    try:
                        st.session_state["raw_ert_datasets"][base_name] = cargar_archivo_tre(u_file)
                        st.success(f"Sección cargada: {base_name}")
                    except Exception as e:
                        st.error(f"Error al leer {u_file.name}: {e}")

    st.divider()
    if st.button("🧪 Generar Línea Sintética de Prueba", use_container_width=True):
        n_sint = len([k for k in st.session_state["raw_ert_datasets"] if k.startswith("Línea_Sintética")]) + 1
        nombre_sint = f"Línea_Sintética_{n_sint}"
        st.session_state["raw_ert_datasets"][nombre_sint] = generar_tre_sintetica()
        st.rerun()

    secciones_disponibles = list(st.session_state["raw_ert_datasets"].keys())
    if secciones_disponibles:
        st.divider()
        seccion_activa = st.selectbox("Sección activa para procesar:", secciones_disponibles)
    else:
        seccion_activa = None

if not seccion_activa:
    st.info(
        "👈 Sube tu archivo **.stg** en la barra lateral "
        "o haz clic en **'Generar Línea Sintética de Prueba'** para iniciar."
    )
    st.stop()

data_raw = st.session_state["raw_ert_datasets"][seccion_activa]

# ==============================================================================
# PESTAÑAS DEL FLUJO DE TRABAJO
# ==============================================================================
tab_qc, tab_inv, tab_3d = st.tabs([
    "🔍 1. Control de Calidad y Georreferenciación",
    "⚙️ 2. Inversión 2D (Asistente Geofísico y Optimización RMS)",
    "🌐 3. Visor 3D de Secciones (Fence Diagram) y Exportación",
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: CONTROL DE CALIDAD, AUTO-LIMPIEZA Y COMPARACIÓN VISUAL
# ------------------------------------------------------------------------------
with tab_qc:
    col_qc, col_geo = st.columns([1.3, 1])

    elec_pos = np.array([[p.x(), p.y(), p.z()] for p in data_raw.sensorPositions()])
    elec_d = elec_pos[:, 0]
    elec_z = elec_pos[:, 1] if np.ptp(elec_pos[:, 1]) > np.ptp(elec_pos[:, 2]) else elec_pos[:, 2]

    rhoa_orig = np.array(data_raw["rhoa"])
    rhoa_min_raw = float(np.floor(np.min(rhoa_orig) * 100.0) / 100.0)
    rhoa_max_raw = float(np.ceil(np.max(rhoa_orig) * 100.0) / 100.0)
    rhoa_med_raw = float(np.median(rhoa_orig))

    rhoa_min_auto, rhoa_max_auto, n_auto_rem = calcular_autolimpieza_iqr(rhoa_orig, factor_iqr=1.5)
    pct_auto_rem = (n_auto_rem / len(rhoa_orig)) * 100.0 if len(rhoa_orig) > 0 else 0.0

    key_rmin = f"rmin_{seccion_activa}"
    key_rmax = f"rmax_{seccion_activa}"
    key_err = f"err_{seccion_activa}"

    if key_rmin not in st.session_state:
        st.session_state[key_rmin] = rhoa_min_raw
    if key_rmax not in st.session_state:
        st.session_state[key_rmax] = rhoa_max_raw
    if key_err not in st.session_state:
        st.session_state[key_err] = 3.0

    with col_qc:
        st.subheader(f"Estadísticas Crudas Originales: {seccion_activa}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Electrodos", data_raw.sensorCount())
        m2.metric("Cuadrupolos (Lecturas)", data_raw.size())
        m3.metric("Longitud de Línea", f"{elec_d.max() - elec_d.min():.1f} m")

        o1, o2, o3 = st.columns(3)
        o1.metric("ρa Mínima Original", f"{np.min(rhoa_orig):.2f} Ω·m")
        o2.metric("ρa Máxima Original", f"{np.max(rhoa_orig):.2f} Ω·m")
        o3.metric("ρa Mediana Original", f"{rhoa_med_raw:.2f} Ω·m")

        st.info(
            f"💡 **Diagnóstico de Auto-Limpieza (IQR Log-Normal):** Se sugiere acotar entre "
            f"**{rhoa_min_auto:.2f} Ω·m** y **{rhoa_max_auto:.2f} Ω·m** "
            f"(descarta **{n_auto_rem}** lecturas atípicas [{pct_auto_rem:.1f}%] y conserva el "
            f"**{100.0 - pct_auto_rem:.1f}%** de tus datos)."
        )

        btn_c1, btn_c2 = st.columns(2)
        if btn_c1.button("✨ Aplicar Auto-Limpieza Sugerida", type="primary", use_container_width=True, key=f"btn_auto_{seccion_activa}"):
            st.session_state[key_rmin] = rhoa_min_auto
            st.session_state[key_rmax] = rhoa_max_auto
            st.session_state[key_err] = 4.5
            st.session_state["residual_masks"].pop(seccion_activa, None)
            st.rerun()

        if btn_c2.button("🔄 Restaurar Datos Originales", use_container_width=True, key=f"btn_orig_{seccion_activa}"):
            st.session_state[key_rmin] = rhoa_min_raw
            st.session_state[key_rmax] = rhoa_max_raw
            st.session_state[key_err] = 3.0
            st.session_state["residual_masks"].pop(seccion_activa, None)
            st.rerun()

        with st.form(key=f"form_qc_{seccion_activa}"):
            st.markdown("**Ajuste Manual de Filtros Pre-Inversión**")
            c_min, c_max = st.columns(2)
            rhoa_min = c_min.number_input(
                "Resistividad Aparente Mín (Ω·m)",
                min_value=0.0,
                max_value=float(rhoa_max_raw),
                format="%.2f",
                key=key_rmin,
            )
            rhoa_max = c_max.number_input(
                "Resistividad Aparente Máx (Ω·m)",
                min_value=0.01,
                max_value=max(float(rhoa_max_raw) * 2.0, 100000.0),
                format="%.2f",
                key=key_rmax,
            )
            err_est = st.slider(
                "Error relativo base para inversión (%)",
                1.0, 15.0,
                step=0.5,
                key=key_err,
            )
            if st.form_submit_button("Aplicar Ajuste Manual y Actualizar Pseudosección"):
                st.session_state["residual_masks"].pop(seccion_activa, None)

        # Aplicar el filtro de rango y también la poda de residuales si el usuario la activó en Pestaña 2
        data_clean = pg.DataContainerERT(data_raw)
        invalid_mask = (data_clean["rhoa"] < (rhoa_min - 1e-4)) | (data_clean["rhoa"] > (rhoa_max + 1e-4))
        data_clean.markInvalid(invalid_mask)
        data_clean.removeInvalid()

        n_podados_rms = 0
        if seccion_activa in st.session_state["residual_masks"]:
            mask_res_bad = st.session_state["residual_masks"][seccion_activa]
            if len(mask_res_bad) == data_clean.size():
                n_podados_rms = int(np.sum(mask_res_bad))
                data_clean.markInvalid(mask_res_bad)
                data_clean.removeInvalid()

        n_descartadas_actual = int(data_raw.size() - data_clean.size())
        data_clean["err"] = ert.estimateError(
            data_clean, relativeError=err_est / 100.0, absoluteUError=5e-5
        )

    with col_geo:
        st.subheader("Georreferenciación UTM de la Línea")
        idx_sec = secciones_disponibles.index(seccion_activa)
        default_y = 3378695.70 - (idx_sec * 60.0)

        c_x1, c_y1 = st.columns(2)
        x_ini = c_x1.number_input("X UTM Inicio (Este)", value=601630.90, format="%.2f", key=f"x1_{seccion_activa}")
        y_ini = c_y1.number_input("Y UTM Inicio (Norte)", value=default_y, format="%.2f", key=f"y1_{seccion_activa}")

        c_x2, c_y2 = st.columns(2)
        x_fin = c_x2.number_input("X UTM Fin (Este)", value=601703.70, format="%.2f", key=f"x2_{seccion_activa}")
        y_fin = c_y2.number_input("Y UTM Fin (Norte)", value=3378425.60, format="%.2f", key=f"y2_{seccion_activa}")

        z_offset = st.number_input(
            "Elevación Base Z (msnm) [0 si el archivo ya trae cota topográfica]",
            value=0.0 if elec_z.mean() > 100 else 27.0,
            step=1.0,
            key=f"zoff_{seccion_activa}",
        )

        dist_utm = np.hypot(x_fin - x_ini, y_fin - y_ini)
        azimut = (np.degrees(np.arctan2(x_fin - x_ini, y_fin - y_ini)) + 360) % 360
        st.info(f"📏 **Longitud en planta:** {dist_utm:.1f} m | 🧭 **Azimut:** {azimut:.1f}°")

        if seccion_activa in st.session_state["crs_datasets"]:
            df_crs = st.session_state["crs_datasets"][seccion_activa]
            if not df_crs.empty:
                with st.expander("📊 Ver Resistencia de Contacto (.crs)", expanded=True):
                    fig_crs = go.Figure(
                        go.Bar(
                            x=df_crs.iloc[:, 0],
                            y=df_crs.iloc[:, -1],
                            marker_color="teal",
                            name="Resistencia de Contacto",
                        )
                    )
                    fig_crs.update_layout(
                        height=220,
                        xaxis_title="Electrodo / Par",
                        yaxis_title="Resistencia (Ω)",
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig_crs, use_container_width=True)

    st.divider()
    st.subheader("📊 Comparativa de Pseudosecciones: Datos Originales vs. Datos Filtrados")

    c_stat1, c_stat2, c_stat3 = st.columns(3)
    c_stat1.metric("Lecturas Originales", f"{data_raw.size()}")
    c_stat2.metric("Lecturas Válidas Activas", f"{data_clean.size()}")
    texto_podados = f" (incluye {n_podados_rms} por poda RMS)" if n_podados_rms > 0 else ""
    c_stat3.metric(
        "Lecturas Descartadas Total",
        f"{n_descartadas_actual} ({(n_descartadas_actual / max(1, data_raw.size())) * 100:.1f}%){texto_podados}"
    )

    modo_vista_pseudo = st.radio(
        "Disposición visual de las pseudosecciones:",
        options=["Lado a Lado (2 Columnas)", "Apiladas Verticalmente (Alineación exacta en distancia)"],
        index=0,
        horizontal=True,
    )

    if modo_vista_pseudo.startswith("Lado a Lado"):
        p_col1, p_col2 = st.columns(2)
    else:
        p_col1 = st.container()
        p_col2 = st.container()

    with p_col1:
        fig_raw, ax_raw = plt.subplots(figsize=(8, 3.3))
        ert.show(data_raw, ax=ax_raw, cMap="Spectral_r", label="ρa Original (Ω·m)")
        ax_raw.set_title(
            f"1. ORIGINAL (Sin filtro): {data_raw.size()} lecturas [{np.min(rhoa_orig):.1f} - {np.max(rhoa_orig):.1f} Ω·m]"
        )
        st.pyplot(fig_raw)
        plt.close(fig_raw)

    with p_col2:
        fig_clean, ax_clean = plt.subplots(figsize=(8, 3.3))
        rhoa_clean_arr = np.array(data_clean["rhoa"])
        ert.show(data_clean, ax=ax_clean, cMap="Spectral_r", label="ρa Filtrada (Ω·m)")
        ax_clean.set_title(
            f"2. ACTUALIZADA (Filtrada): {data_clean.size()} lecturas [{np.min(rhoa_clean_arr):.1f} - {np.max(rhoa_clean_arr):.1f} Ω·m]"
        )
        st.pyplot(fig_clean)
        plt.close(fig_clean)


# ------------------------------------------------------------------------------
# PESTAÑA 2: INVERSIÓN 2D CON ASISTENTE GEOFÍSICO Y OPTIMIZADOR DE RMS
# ------------------------------------------------------------------------------
with tab_inv:
    # 1. Ejecutar diagnóstico geofísico experto sobre los datos activos
    diag = diagnosticar_parametros_geofisicos(data_clean, elec_d)

    opciones_esquema = [
        "Suavizada L2 (Occam / Smooth)",
        "Robusta en Datos L1 (Ignora outliers)",
        "Bloques / Contactos Abruptos (Blocky L1)",
    ]
    opciones_dx = [0.5, 0.33, 0.25]

    # Inicializar llaves de estado de los controles de inversión para esta sección
    k_lam = f"inv_lam_{seccion_activa}"
    k_zw = f"inv_zw_{seccion_activa}"
    k_esq = f"inv_esq_{seccion_activa}"
    k_dx = f"inv_dx_{seccion_activa}"
    k_dep = f"inv_dep_{seccion_activa}"
    k_iter = f"inv_iter_{seccion_activa}"
    k_cov = f"inv_cov_{seccion_activa}"
    k_auto_run = f"auto_run_flag_{seccion_activa}"

    if k_lam not in st.session_state:
        st.session_state[k_lam] = diag["lam_sug"]
    if k_zw not in st.session_state:
        st.session_state[k_zw] = diag["z_weight_sug"]
    if k_esq not in st.session_state:
        st.session_state[k_esq] = opciones_esquema[diag["esquema_idx"]]
    if k_dx not in st.session_state:
        st.session_state[k_dx] = diag["para_dx_sug"]
    if k_dep not in st.session_state:
        st.session_state[k_dep] = diag["prof_sug"]
    if k_iter not in st.session_state:
        st.session_state[k_iter] = diag["iter_sug"]
    if k_cov not in st.session_state:
        st.session_state[k_cov] = diag["cov_sug"]

    col_params, col_plot = st.columns([1.05, 2.3])

    with col_params:
        st.subheader("🧠 Asistente Geofísico de Inversión")
        st.info(
            f"**Diagnóstico del Tendido (`{seccion_activa}`):**\n"
            f"- **Geometría:** $L = {diag['L']:.0f}\\text{{ m}}$, espaciamiento $a = {diag['a_min']:.1f}\\text{{ m}}$ "
            f"→ **DOI Óptimo: {diag['prof_sug']:.0f} m**.\n"
            f"- **Ambiente inferido:** {diag['ambiente_txt']} (mediana $\\rho_a = {diag['mediana_ohm']:.1f}\\ \\Omega\\cdot\\text{{m}}$) "
            f"→ **Anisotropía `zWeight` = {diag['z_weight_sug']:.2f}**.\n"
            f"- **Criterio numérico:** {diag['motivo_esquema']} (**$\\lambda = {diag['lam_sug']:.0f}$**)."
        )

        if st.button(
            "✨ Aplicar Configuración Geofísica Sugerida e Invertir",
            type="primary",
            use_container_width=True,
            key=f"btn_apply_geo_{seccion_activa}",
        ):
            # Guardar modelo actual como previo antes de actualizar para poder comparar
            if seccion_activa in st.session_state["modelos_geofisicos"]:
                st.session_state["modelos_previos"][seccion_activa] = (
                    st.session_state["modelos_geofisicos"][seccion_activa].copy()
                )
            st.session_state[k_lam] = diag["lam_sug"]
            st.session_state[k_zw] = diag["z_weight_sug"]
            st.session_state[k_esq] = opciones_esquema[diag["esquema_idx"]]
            st.session_state[k_dx] = diag["para_dx_sug"]
            st.session_state[k_dep] = diag["prof_sug"]
            st.session_state[k_iter] = diag["iter_sug"]
            st.session_state[k_cov] = diag["cov_sug"]
            st.session_state[k_auto_run] = True
            st.rerun()

        with st.form(key=f"form_inv_{seccion_activa}"):
            st.markdown("**Ajuste Manual de Parámetros de Inversión**")
            lam = st.slider("Regularización (λ - Suavizado)", 1.0, 150.0, step=1.0, key=k_lam)
            z_weight = st.slider("Anisotropía Estructural (zWeight)", 0.05, 1.0, step=0.05, key=k_zw)
            tipo_inv = st.radio(
                "Esquema de Inversión (Estilo EarthImager)",
                options=opciones_esquema,
                key=k_esq,
            )
            c_m1, c_m2 = st.columns(2)
            para_dx = c_m1.selectbox("Refinamiento de celda", opciones_dx, key=k_dx)
            max_depth = c_m2.number_input("Prof. Máx (m)", min_value=0.0, step=2.0, key=k_dep)
            max_iter = st.slider("Iteraciones máximas", 3, 25, key=k_iter)
            cov_threshold = st.slider("Umbral DOI (Sensibilidad)", -3.5, 0.0, step=0.1, key=k_cov)

            btn_invertir = st.form_submit_button("🚀 Ejecutar Inversión Manual", use_container_width=True)

        st.divider()
        st.subheader("🎨 Opciones de Visualización (Ω·m)")
        eje_vertical_modo = st.radio(
            "Eje Vertical de la Sección:",
            options=[
                "Profundidad bajo terreno (m) [0 a -Prof]",
                "Elevación Topográfica (msnm)",
            ],
            index=0,
        )
        exag_2d = st.slider("Exageración Vertical 2D", 0.5, 4.0, 1.8, 0.1)
        usar_recorte_doi = st.checkbox(
            "Recortar zonas de baja sensibilidad (DOI)",
            value=False,
            help="Desactivado muestra el trapecio completo hasta la profundidad máxima (Estilo EarthImager)."
        )
        modo_escala = st.radio(
            "Distribución de Color (Etiquetas siempre en Ω·m):",
            options=[
                "Proporcional Logarítmica (Recomendada en Geofísica)",
                "Lineal Directa (Escala aritmética en Ω·m)",
            ],
            index=0,
        )
        mostrar_isolineas = st.checkbox("Mostrar líneas de contorno (Isolíneas)", value=True)
        paleta_color = st.selectbox("Paleta de colores", ["Turbo", "Jet", "Spectral_r", "Viridis"], index=0)

    robust_data = tipo_inv in ["Robusta en Datos L1 (Ignora outliers)", "Bloques / Contactos Abruptos (Blocky L1)"]
    blocky_model = tipo_inv == "Bloques / Contactos Abruptos (Blocky L1)"

    disparar_inversion = (
        btn_invertir
        or st.session_state.get(k_auto_run, False)
        or (seccion_activa in st.session_state["modelos_geofisicos"])
    )

    if btn_invertir and (seccion_activa in st.session_state["modelos_geofisicos"]):
        st.session_state["modelos_previos"][seccion_activa] = (
            st.session_state["modelos_geofisicos"][seccion_activa].copy()
        )

    if disparar_inversion:
        st.session_state[k_auto_run] = False
        with st.spinner(f"Invirtiendo {seccion_activa} por elementos finitos ({data_clean.size()} cuadrupolos)..."):
            (
                nodes_2d, triangles, cell_res, cell_cov,
                node_res, node_cov, centers_2d, res_raw, cov_raw,
                rhoa_obs, rhoa_calc, res_pct, metrics
            ) = ejecutar_inversion_tre(
                data_clean, seccion_activa, data_clean.size(),
                lam, z_weight, robust_data, blocky_model, max_iter, para_dx, max_depth
            )

            nodes_utm = proyectar_2d_a_utm(nodes_2d, x_ini, y_ini, x_fin, y_fin, z_offset)
            centers_utm = proyectar_2d_a_utm(centers_2d, x_ini, y_ini, x_fin, y_fin, z_offset)
            elecs_utm = proyectar_2d_a_utm(np.column_stack([elec_d, elec_z]), x_ini, y_ini, x_fin, y_fin, z_offset)

            st.session_state["modelos_geofisicos"][seccion_activa] = {
                "metodo": "TRE_2D",
                "nodes_2d": nodes_2d,
                "nodes_utm": nodes_utm,
                "triangles": triangles,
                "cell_res": cell_res,
                "cell_cov": cell_cov,
                "node_res": node_res,
                "node_cov": node_cov,
                "centers_2d": centers_2d,
                "centers_utm": centers_utm,
                "centers_res": res_raw,
                "centers_cov": cov_raw,
                "elecs_d": elec_d,
                "elecs_z": elec_z,
                "elecs_utm": elecs_utm,
                "z_offset": z_offset,
                "cov_threshold": cov_threshold,
                "rhoa_obs": rhoa_obs,
                "rhoa_calc": rhoa_calc,
                "res_pct": res_pct,
                "esquema_txt": tipo_inv,
                "metrics": metrics,
            }

        with col_plot:
            mod = st.session_state["modelos_geofisicos"][seccion_activa]
            prof_alcanzada = float(elec_z.max() - mod["nodes_2d"][:, 1].min())
            rms_actual = mod["metrics"]["rms_rel"]
            chi2_actual = mod["metrics"]["chi2"]
            r2_actual = mod["metrics"]["r2_log"]

            # Comprobar si existe un modelo previo para mostrar deltas de mejora
            tiene_previo = seccion_activa in st.session_state["modelos_previos"]
            delta_rms = None
            delta_chi2 = None
            if tiene_previo:
                mod_prev = st.session_state["modelos_previos"][seccion_activa]
                diff_r = rms_actual - mod_prev["metrics"]["rms_rel"]
                diff_c = chi2_actual - mod_prev["metrics"]["chi2"]
                if abs(diff_r) > 0.01:
                    delta_rms = f"{diff_r:+.2f} %"
                    delta_chi2 = f"{diff_c:+.2f}"

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Error RMS Relativo", f"{rms_actual:.2f} %", delta=delta_rms, delta_color="inverse")
            k2.metric("Ajuste χ²", f"{chi2_actual:.2f}", delta=delta_chi2, delta_color="inverse")
            k3.metric("Correlación R² (Obs vs Calc)", f"{r2_actual:.3f}")
            k4.metric("Profundidad Malla", f"{prof_alcanzada:.1f} m ({mod['metrics']['n_data']} lect.)")

            # ------------------------------------------------------------------
            # SEMÁFORO DE CALIDAD POST-INVERSIÓN Y PODA DE RESIDUALES (RMS TRIMMER)
            # ------------------------------------------------------------------
            abs_res = np.abs(mod["res_pct"])
            umbral_poda_pct = 25.0
            mask_rebeldes = abs_res > umbral_poda_pct
            n_rebeldes = int(np.sum(mask_rebeldes))

            if rms_actual <= 7.5:
                st.success(
                    f"✅ **Calidad Geofísica Óptima (RMS = {rms_actual:.2f}%, R² = {r2_actual:.3f}):** "
                    "El modelo converge dentro del margen de ruido instrumental sin sobreajustar artefactos."
                )
            elif rms_actual <= 12.0:
                st.warning(
                    f"⚠️ **Calidad Aceptable con Ruido Moderado (RMS = {rms_actual:.2f}%):** "
                    f"Se detectaron **{n_rebeldes}** cuadrupolos con residual > {umbral_poda_pct:.0f}% que elevan el RMS."
                )
            else:
                st.error(
                    f"🚨 **RMS Elevado ({rms_actual:.2f}%):** Existen **{n_rebeldes}** lecturas discordantes "
                    f"(residual > {umbral_poda_pct:.0f}%) que impiden un ajuste suave de la ecuación de Poisson."
                )

            if n_rebeldes > 0 and rms_actual > 6.0:
                c_opt1, c_opt2 = st.columns([1.8, 1])
                if c_opt1.button(
                    f"🎯 Auto-Optimizar RMS: Podar {n_rebeldes} Lecturas Discordantes (> {umbral_poda_pct:.0f}% error) y Re-Invertir",
                    type="primary",
                    use_container_width=True,
                    key=f"btn_trim_rms_{seccion_activa}",
                ):
                    st.session_state["modelos_previos"][seccion_activa] = mod.copy()
                    st.session_state["residual_masks"][seccion_activa] = mask_rebeldes
                    st.session_state[k_esq] = "Robusta en Datos L1 (Ignora outliers)"
                    st.session_state[k_lam] = max(12.0, float(lam) * 0.85)
                    st.session_state[k_auto_run] = True
                    st.rerun()

                if seccion_activa in st.session_state["residual_masks"]:
                    if c_opt2.button(
                        "↩️ Deshacer Poda de Residuales",
                        use_container_width=True,
                        key=f"btn_untrim_{seccion_activa}",
                    ):
                        st.session_state["residual_masks"].pop(seccion_activa, None)
                        st.session_state[k_auto_run] = True
                        st.rerun()

            # Controles de escala de color en Ω·m reales
            res_valid_ohm = mod["centers_res"]
            r_min_inv = float(np.floor(np.min(res_valid_ohm) * 10.0) / 10.0)
            r_max_inv = float(np.ceil(np.max(res_valid_ohm) * 10.0) / 10.0)

            c_r1, c_r2 = st.columns(2)
            r_min_user = c_r1.number_input(
                "Mínimo Barra de Color (Ω·m)",
                value=max(0.1, r_min_inv),
                step=1.0,
                key=f"cmin2d_{seccion_activa}",
            )
            r_max_user = c_r2.number_input(
                "Máximo Barra de Color (Ω·m)",
                value=max(10.0, r_max_inv),
                step=10.0,
                key=f"cmax2d_{seccion_activa}",
            )

            # Interpolación de la sección actual
            d_grid, z_grid_msnm, rho_grid, log_rho_grid = interpolar_malla_suave_2d(
                mod["centers_2d"], mod["centers_res"], mod["centers_cov"],
                elec_d, elec_z, z_offset, cov_threshold, usar_recorte_doi=usar_recorte_doi
            )

            cota_superficie = float(np.mean(elec_z + z_offset))
            if eje_vertical_modo.startswith("Profundidad"):
                y_grid_plot = z_grid_msnm - cota_superficie
                y_elecs_plot = (elec_z + z_offset) - cota_superficie
                y_label = "Profundidad (m)"
            else:
                y_grid_plot = z_grid_msnm
                y_elecs_plot = elec_z + z_offset
                y_label = "Elevación (msnm)"

            # ------------------------------------------------------------------
            # COMPARATIVA VISUAL: MODELO ANTERIOR VS. MODELO ACTUALIZADO
            # ------------------------------------------------------------------
            ver_comparativa_inv = False
            if tiene_previo:
                ver_comparativa_inv = st.checkbox(
                    "🔍 Comparar Modelo Anterior vs. Modelo Actualizado (Antes / Después)",
                    value=True,
                    key=f"chk_comp_inv_{seccion_activa}",
                )

            if ver_comparativa_inv and tiene_previo:
                mod_prev = st.session_state["modelos_previos"][seccion_activa]
                d_grid_p, z_grid_p, rho_grid_p, log_rho_grid_p = interpolar_malla_suave_2d(
                    mod_prev["centers_2d"], mod_prev["centers_res"], mod_prev["centers_cov"],
                    elec_d, elec_z, z_offset, cov_threshold, usar_recorte_doi=usar_recorte_doi
                )
                y_grid_prev = (z_grid_p - cota_superficie) if eje_vertical_modo.startswith("Profundidad") else z_grid_p

                fig_prev = crear_figura_plotly_2d(
                    d_grid_p, y_grid_prev, rho_grid_p, log_rho_grid_p, elec_d, y_elecs_plot,
                    r_min_user, r_max_user, modo_escala, paleta_color, mostrar_isolineas,
                    y_label, exag_2d,
                    f"1. MODELO ANTERIOR — RMS: {mod_prev['metrics']['rms_rel']:.2f}% | χ²: {mod_prev['metrics']['chi2']:.1f} ({mod_prev['metrics']['n_data']} lect.)"
                )
                st.plotly_chart(fig_prev, use_container_width=True)

            fig_smooth = crear_figura_plotly_2d(
                d_grid, y_grid_plot, rho_grid, log_rho_grid, elec_d, y_elecs_plot,
                r_min_user, r_max_user, modo_escala, paleta_color, mostrar_isolineas,
                y_label, exag_2d,
                f"2. MODELO ACTUALIZADO ({prof_alcanzada:.0f} m prof.) — RMS: {rms_actual:.2f}% | χ²: {chi2_actual:.1f} ({mod['metrics']['n_data']} lect.)"
                if (ver_comparativa_inv and tiene_previo)
                else f"Sección Geoeléctrica Invertida ({prof_alcanzada:.0f} m de prof.) — {seccion_activa} [RMS: {rms_actual:.2f}%]"
            )
            st.plotly_chart(fig_smooth, use_container_width=True)

            # Panel de Diagnóstico de Residuales (Crossplot Observado vs Calculado)
            with st.expander("📈 Diagnóstico de Ajuste Geofísico (ρa Observada vs. Calculada y Residuales)", expanded=False):
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    fig_cross = go.Figure()
                    fig_cross.add_trace(
                        go.Scatter(
                            x=mod["rhoa_obs"],
                            y=mod["rhoa_calc"],
                            mode="markers",
                            marker=dict(
                                size=6,
                                color=np.abs(mod["res_pct"]),
                                colorscale="YlOrRd",
                                cmin=0,
                                cmax=35,
                                showscale=True,
                                colorbar=dict(title="|Error| (%)"),
                            ),
                            name="Cuadrupolos",
                            hovertemplate="ρa Obs: %{x:.1f} Ω·m<br>ρa Calc: %{y:.1f} Ω·m<extra></extra>",
                        )
                    )
                    min_line = float(min(mod["rhoa_obs"].min(), mod["rhoa_calc"].min()))
                    max_line = float(max(mod["rhoa_obs"].max(), mod["rhoa_calc"].max()))
                    fig_cross.add_trace(
                        go.Scatter(
                            x=[min_line, max_line],
                            y=[min_line, max_line],
                            mode="lines",
                            line=dict(color="black", dash="dash"),
                            name="Ajuste 1:1 Ideal",
                        )
                    )
                    fig_cross.update_layout(
                        title=f"Curva de Ajuste Observado vs. Calculado (R² = {r2_actual:.3f})",
                        xaxis=dict(title="ρa Observada de Campo (Ω·m)", type="log"),
                        yaxis=dict(title="ρa Calculada por el Modelo (Ω·m)", type="log"),
                        height=340,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_cross, use_container_width=True)

                with col_res2:
                    fig_hist = go.Figure(
                        go.Histogram(
                            x=mod["res_pct"],
                            nbinsx=35,
                            marker_color="steelblue",
                            name="Residuales (%)",
                        )
                    )
                    fig_hist.add_vline(x=-umbral_poda_pct, line_dash="dot", line_color="red")
                    fig_hist.add_vline(x=umbral_poda_pct, line_dash="dot", line_color="red")
                    fig_hist.update_layout(
                        title="Distribución de Error Residual por Cuadrupolo (%)",
                        xaxis_title="Error Residual [(Calc - Obs) / Obs] (%)",
                        yaxis_title="Frecuencia (N° de lecturas)",
                        height=340,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

            with st.expander("📄 Ver / Descargar Figura para Informe Técnico (Renderizado Gouraud)", expanded=False):
                fig_pub, ax_pub = plt.subplots(figsize=(10, 3.6), dpi=200)
                cf = ax_pub.contourf(
                    d_grid, y_grid_plot, rho_grid,
                    levels=np.geomspace(r_min_user, r_max_user, 24)
                    if modo_escala.startswith("Proporcional")
                    else np.linspace(r_min_user, r_max_user, 24),
                    cmap="turbo" if paleta_color == "Turbo" else "jet",
                    extend="both",
                )
                if mostrar_isolineas:
                    ax_pub.contour(d_grid, y_grid_plot, rho_grid, levels=cf.levels[::2], colors="k", linewidths=0.35, alpha=0.5)
                ax_pub.plot(elec_d, y_elecs_plot, "kv", markersize=4, label="Electrodos")
                ax_pub.set_xlabel("Distancia (m)")
                ax_pub.set_ylabel(y_label)
                ax_pub.set_title(f"Tomografía de Resistividad Eléctrica: {seccion_activa} (RMS: {rms_actual:.1f}%)")
                ax_pub.set_aspect(exag_2d)
                cbar = fig_pub.colorbar(cf, ax=ax_pub, pad=0.02)
                cbar.set_label("Resistividad (Ω·m)")
                cbar.formatter = ticker.FormatStrFormatter("%.0f")
                cbar.update_ticks()
                st.pyplot(fig_pub)
                plt.close(fig_pub)


# ------------------------------------------------------------------------------
# PESTAÑA 3: VISOR 3D MULTISECCIÓN (FENCE DIAGRAM SUAVIZADO EN Ω·m)
# ------------------------------------------------------------------------------
with tab_3d:
    modelos_tre = {
        k: v for k, v in st.session_state["modelos_geofisicos"].items() if v["metodo"] == "TRE_2D"
    }

    if not modelos_tre:
        st.warning("Primero invierte al menos una sección en la Pestaña 2 para visualizarla en 3D.")
    else:
        col_ctrl3d, col_vis3d = st.columns([1, 3])

        with col_ctrl3d:
            st.subheader("Control de Escena 3D")
            secciones_a_mostrar = st.multiselect(
                "Secciones activas en el cubo 3D:",
                options=list(modelos_tre.keys()),
                default=list(modelos_tre.keys()),
            )
            exag_vertical = st.slider("Exageración Vertical (Z)", 0.5, 5.0, 1.5, 0.25)

            c3_min, c3_max = st.columns(2)
            rho3d_min = c3_min.number_input("Mín 3D (Ω·m)", value=2.0, min_value=0.1, step=1.0)
            rho3d_max = c3_max.number_input("Máx 3D (Ω·m)", value=350.0, min_value=5.0, step=10.0)

            suavizado_3d = st.checkbox("Suavizar colores en malla 3D (Gouraud)", value=True)
            mostrar_electrodos = st.checkbox("Mostrar trazas de electrodos", value=True)

            st.divider()
            st.subheader("Exportación de Resultados")
            dfs_export = []
            for nom, m in modelos_tre.items():
                mask_c = m["centers_cov"] >= m["cov_threshold"]
                df_sec = pd.DataFrame({
                    "Seccion": nom,
                    "Metodo": m["metodo"],
                    "X_UTM": m["centers_utm"][mask_c, 0],
                    "Y_UTM": m["centers_utm"][mask_c, 1],
                    "Z_msnm": m["centers_utm"][mask_c, 2],
                    "Resistividad_Ohm_m": m["centers_res"][mask_c],
                    "Cobertura_DOI": m["centers_cov"][mask_c],
                })
                dfs_export.append(df_sec)

            if dfs_export:
                df_total = pd.concat(dfs_export, ignore_index=True)
                csv_bytes = df_total.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Descargar Nube 3D Georreferenciada (.CSV)",
                    data=csv_bytes,
                    file_name="secciones_TRE_georreferenciadas_UTM.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col_vis3d:
            fig_3d = go.Figure()
            tick_vals_3d, tick_text_3d = generar_ticks_ohm_m(rho3d_min, rho3d_max)

            for idx, nom in enumerate(secciones_a_mostrar):
                m = modelos_tre[nom]
                mask_doi = m["cell_cov"] >= m["cov_threshold"]
                tris_v = m["triangles"][mask_doi]

                if suavizado_3d:
                    intensidad = np.log10(np.clip(m["node_res"], 0.1, 1e5))
                    modo_int = "vertex"
                    vals_reales_hover = m["node_res"]
                else:
                    intensidad = np.log10(np.clip(m["cell_res"][mask_doi], 0.1, 1e5))
                    modo_int = "cell"
                    vals_reales_hover = m["node_res"]

                fig_3d.add_trace(
                    go.Mesh3d(
                        x=m["nodes_utm"][:, 0],
                        y=m["nodes_utm"][:, 1],
                        z=m["nodes_utm"][:, 2],
                        i=tris_v[:, 0],
                        j=tris_v[:, 1],
                        k=tris_v[:, 2],
                        intensity=intensidad,
                        intensitymode=modo_int,
                        customdata=vals_reales_hover,
                        cmin=np.log10(rho3d_min),
                        cmax=np.log10(rho3d_max),
                        colorscale="Turbo",
                        showscale=(idx == 0),
                        colorbar=dict(
                            title="Resistividad (Ω·m)",
                            tickvals=tick_vals_3d,
                            ticktext=tick_text_3d,
                        ),
                        name=nom,
                        hovertemplate=(
                            f"<b>{nom}</b><br>"
                            "X UTM: %{x:.1f} m<br>"
                            "Y UTM: %{y:.1f} m<br>"
                            "Elevación Z: %{z:.1f} msnm<br>"
                            "<b>Resistividad:</b> %{customdata:.1f} Ω·m<extra></extra>"
                        ),
                    )
                )

                if mostrar_electrodos:
                    fig_3d.add_trace(
                        go.Scatter3d(
                            x=m["elecs_utm"][:, 0],
                            y=m["elecs_utm"][:, 1],
                            z=m["elecs_utm"][:, 2],
                            mode="markers+lines",
                            marker=dict(size=3, color="black"),
                            line=dict(color="black", width=2),
                            name=f"Electrodos ({nom})",
                        )
                    )

            fig_3d.update_layout(
                title="Diagrama de Valla 3D (Fence Diagram Suavizado en Ω·m)",
                height=650,
                scene=dict(
                    xaxis_title="Este X (UTM m)",
                    yaxis_title="Norte Y (UTM m)",
                    zaxis_title="Elevación Z (msnm)",
                    aspectmode="manual",
                    aspectratio=dict(x=1.5, y=1.5, z=0.6 * exag_vertical),
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_3d, use_container_width=True)
