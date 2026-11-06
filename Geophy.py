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

if "raw_ert_datasets" not in st.session_state:
    st.session_state["raw_ert_datasets"] = {}

if "crs_datasets" not in st.session_state:
    st.session_state["crs_datasets"] = {}


# ==============================================================================
# UTILIDADES PARA ESCALAS DE COLOR Y AUTO-LIMPIEZA ESTADÍSTICA
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

    # Límites estadísticos Tukey en escala logarítmica
    low_log = q25 - (factor_iqr * iqr)
    high_log = q75 + (factor_iqr * iqr)

    r_min_sug = max(float(np.min(rhoa_pos)), float(10 ** low_log))
    r_max_sug = min(float(np.max(rhoa_pos)), float(10 ** high_log))

    r_min_sug = float(np.floor(r_min_sug * 100.0) / 100.0)
    r_max_sug = float(np.ceil(r_max_sug * 100.0) / 100.0)

    mask_out = (rhoa_array < r_min_sug) | (rhoa_array > r_max_sug)
    n_removidos = int(np.sum(mask_out))

    return r_min_sug, r_max_sug, n_removidos


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
    Ejecuta la inversión 2D en PyGIMLi y devuelve resistividad por celda y por vértice.
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

    metrics = {
        "rms_rel": float(mgr.inv.relrms()),
        "chi2": float(mgr.inv.chi2()),
        "n_cells": mesh_2d.cellCount(),
        "n_data": int(n_valid),
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


# ==============================================================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ==============================================================================
st.title("⚡ Módulo 1: Procesamiento e Inversión 2D/3D de TRE")
st.markdown(
    "Carga archivos crudos de resistividad eléctrica (**AGI SuperSting .stg / .crs**, "
    "**RES2DINV .dat**, **.ohm**), aplica auto-limpieza o filtrado manual, invierte con elementos finitos "
    "y visualiza secciones suavizadas en notación normal (**Ω·m**)."
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
    "⚙️ 2. Inversión 2D (Estilo Libro / EarthImager)",
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

    # 1. Valores originales exactos del archivo crudo
    rhoa_orig = np.array(data_raw["rhoa"])
    rhoa_min_raw = float(np.floor(np.min(rhoa_orig) * 100.0) / 100.0)
    rhoa_max_raw = float(np.ceil(np.max(rhoa_orig) * 100.0) / 100.0)
    rhoa_med_raw = float(np.median(rhoa_orig))

    # 2. Calcular propuesta de Auto-Limpieza estadística (IQR sobre Log10)
    rhoa_min_auto, rhoa_max_auto, n_auto_rem = calcular_autolimpieza_iqr(rhoa_orig, factor_iqr=1.5)
    pct_auto_rem = (n_auto_rem / len(rhoa_orig)) * 100.0 if len(rhoa_orig) > 0 else 0.0

    # 3. Inicializar en session_state con los valores originales cuando se sube el archivo
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

        # Recuadro de sugerencia de Auto-Limpieza con botones de acción rápida
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
            st.rerun()

        if btn_c2.button("🔄 Restaurar Datos Originales", use_container_width=True, key=f"btn_orig_{seccion_activa}"):
            st.session_state[key_rmin] = rhoa_min_raw
            st.session_state[key_rmax] = rhoa_max_raw
            st.session_state[key_err] = 3.0
            st.rerun()

        # Controles manuales (siempre disponibles por si el usuario prefiere ajustar a mano)
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
            st.form_submit_button("Aplicar Ajuste Manual y Actualizar Pseudosección")

        # Aplicar el filtro activo a una copia limpia
        data_clean = pg.DataContainerERT(data_raw)
        invalid_mask = (data_clean["rhoa"] < (rhoa_min - 1e-4)) | (data_clean["rhoa"] > (rhoa_max + 1e-4))
        n_descartadas_actual = int(np.sum(invalid_mask))
        data_clean.markInvalid(invalid_mask)
        data_clean.removeInvalid()
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

    # --------------------------------------------------------------------------
    # COMPARATIVA DE PSEUDOSECCIONES: ORIGINAL (CRUDA) VS. ACTUALIZADA (FILTRADA)
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Comparativa de Pseudosecciones: Datos Originales vs. Datos Filtrados")

    c_stat1, c_stat2, c_stat3 = st.columns(3)
    c_stat1.metric("Lecturas Originales", f"{data_raw.size()}")
    c_stat2.metric("Lecturas Válidas Activas", f"{data_clean.size()}")
    c_stat3.metric(
        "Lecturas Descartadas por el Filtro",
        f"{n_descartadas_actual} ({(n_descartadas_actual / max(1, data_raw.size())) * 100:.1f}%)"
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
# PESTAÑA 2: INVERSIÓN 2D (PROFUNDIDAD COMPLETA Y VALORES EN Ω·m)
# ------------------------------------------------------------------------------
with tab_inv:
    col_params, col_plot = st.columns([1, 2.35])

    with col_params:
        with st.form(key=f"form_inv_{seccion_activa}"):
            st.subheader("Parámetros de Inversión")
            lam = st.slider("Regularización (λ - Suavizado)", 1.0, 150.0, 15.0, 1.0)
            z_weight = st.slider("Anisotropía Estructural (zWeight)", 0.05, 1.0, 0.3, 0.05)
            tipo_inv = st.radio(
                "Esquema de Inversión (Estilo EarthImager)",
                options=[
                    "Suavizada L2 (Occam / Smooth)",
                    "Robusta en Datos L1 (Ignora outliers)",
                    "Bloques / Contactos Abruptos (Blocky L1)",
                ],
                index=1,
            )
            c_m1, c_m2 = st.columns(2)
            para_dx = c_m1.selectbox("Refinamiento de celda", [0.5, 0.33, 0.25], index=1)
            max_depth = c_m2.number_input("Prof. Máx (m)", value=68.0, min_value=0.0, step=5.0)
            max_iter = st.slider("Iteraciones máximas", 3, 20, 10)
            cov_threshold = st.slider("Umbral DOI (Sensibilidad)", -3.5, 0.0, -2.5, 0.1)

            btn_invertir = st.form_submit_button("🚀 Ejecutar Inversión y Actualizar 3D", type="primary")

        st.divider()
        st.subheader("🎨 Opciones de Visualización (Ω·m)")
        eje_vertical_modo = st.radio(
            "Eje Vertical de la Sección:",
            options=[
                "Profundidad bajo terreno (m) [0 a -70 m]",
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

    if btn_invertir or (seccion_activa in st.session_state["modelos_geofisicos"]):
        with st.spinner(f"Invirtiendo {seccion_activa} por elementos finitos..."):
            (
                nodes_2d, triangles, cell_res, cell_cov,
                node_res, node_cov, centers_2d, res_raw, cov_raw, metrics
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
                "metrics": metrics,
            }

        with col_plot:
            mod = st.session_state["modelos_geofisicos"][seccion_activa]
            prof_alcanzada = float(elec_z.max() - mod["nodes_2d"][:, 1].min())
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Error RMS Relativo", f"{mod['metrics']['rms_rel']:.2f} %")
            k2.metric("Ajuste χ²", f"{mod['metrics']['chi2']:.2f}")
            k3.metric("Profundidad Malla", f"{prof_alcanzada:.1f} m")
            k4.metric("Celdas Activas", mod["metrics"]["n_cells"])

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

            fig_smooth = go.Figure()

            if modo_escala.startswith("Proporcional Logarítmica"):
                tick_vals_log, tick_text_ohm = generar_ticks_ohm_m(r_min_user, r_max_user)
                fig_smooth.add_trace(
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
                fig_smooth.add_trace(
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

            fig_smooth.add_trace(
                go.Scatter(
                    x=elec_d,
                    y=y_elecs_plot,
                    mode="markers+lines",
                    marker=dict(symbol="triangle-down", size=7, color="black"),
                    line=dict(color="black", width=1.5),
                    name="Electrodos",
                )
            )

            valid_rows = ~np.all(np.isnan(rho_grid), axis=1)
            y_min_valid = float(np.nanmin(y_grid_plot[valid_rows])) if np.any(valid_rows) else float(np.min(y_grid_plot))
            y_max_valid = float(np.nanmax(y_elecs_plot)) + 3.0

            fig_smooth.update_layout(
                title=f"Sección Geoeléctrica Invertida ({prof_alcanzada:.0f} m de prof.) — {seccion_activa}",
                xaxis_title="Distancia sobre la línea (m)",
                yaxis_title=y_label,
                height=520,
                xaxis=dict(range=[elec_d.min() - 2, elec_d.max() + 2]),
                yaxis=dict(
                    range=[y_min_valid - 2, y_max_valid],
                    scaleanchor="x",
                    scaleratio=exag_2d,
                ),
                margin=dict(l=20, r=20, t=45, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            )
            st.plotly_chart(fig_smooth, use_container_width=True)

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
                ax_pub.set_title(f"Tomografía de Resistividad Eléctrica: {seccion_activa} (RMS: {mod['metrics']['rms_rel']:.1f}%)")
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
