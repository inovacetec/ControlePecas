# app_dxf_comparador.py
import streamlit as st
import ezdxf
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os
import json

# -----------------------
# Config da página
# -----------------------
st.set_page_config(
    page_title="DXF Controle de Peças",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# CSS (Tema Dark com alto contraste)
# -----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --brand: #FF9500; }
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
* { font-family: 'Inter', sans-serif; }

/* Título */
.main-title{
  font-size: 2.2rem; font-weight: 700;
  background: linear-gradient(135deg, var(--brand), #FFB84D);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin: 0 0 1.25rem 0; text-align:center;
}

/* Sidebar dark */
[data-testid="stSidebar"]{
  background: #0d1117 !important; color: #e6edf3 !important;
}
[data-testid="stSidebar"] *{ color:#e6edf3 !important; }

/* Cabeçalhos */
.section-header{
  font-size:1.15rem; font-weight:600; color:#f0f6fc;
  border-bottom: 2px solid var(--brand);
  padding-bottom:.35rem; margin-bottom: .9rem;
  display:flex; align-items:center; gap:.5rem;
}

/* Labels */
.stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stFileUploader label{
  color:#e6edf3 !important; font-weight:500 !important; margin-bottom: .35rem !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select{
  border-radius:12px; border:1.5px solid #30363d;
  padding:.6rem .8rem; color:#f0f6fc !important; background:#161b22 !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus{
  border-color: var(--brand); box-shadow: 0 0 0 3px rgba(255,149,0,.15);
}

/* Uploader (dropzone) */
.stFileUploader [data-testid="stFileUploadDropzone"]{
  border:2px dashed var(--brand) !important;
  background: #161b22;
  border-radius:12px; padding:1rem; color:#e6edf3;
}

/* Ocultar input nativo */
.stFileUploader input[type="file"]{
  opacity:0 !important; width:0 !important; height:0 !important;
  position:absolute !important; z-index:-1 !important; pointer-events:none !important;
}

/* Botões */
.stButton > button{
  background: linear-gradient(135deg, var(--brand), #FFB84D) !important;
  color:#fff !important; font-weight:600 !important;
}

/* Alerts */
.stAlert { border-radius:12px; color:#e6edf3 !important; }

/* Dataframe */
[data-testid="stTable"] th, [data-testid="stTable"] td{
  color:#e6edf3 !important; background:#0d1117 !important;
  border-color:#30363d !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Defaults / Perfis (regex)  **CORRIGIDO: raw strings válidas**
# -----------------------
DEFAULT_PILAR_REGEX = r"\b(?:PILAR|PT|P)\s*[-.]?\s*\d+[A-Z]?(?:=\s*[A-Z]?\d+)*\b"
DEFAULT_VIGA_REGEX  = r"\b(?:VIGA|VT|VB|V)\s*[-.]?\s*\d+[A-Z]?(?:=\s*[A-Z]?\d+)*\b"
DEFAULT_SECAO_REGEX = r"\b\d{1,3}\s*[/xX×]\s*\d{1,3}\b"

DEFAULT_PROFILES = {
    "Pilar (PT1A e 19x199)": {"pilar": DEFAULT_PILAR_REGEX, "secao": DEFAULT_SECAO_REGEX, "radius": 400.0},
    "Viga (V1 e 19/19)"   : {"pilar": DEFAULT_VIGA_REGEX , "secao": DEFAULT_SECAO_REGEX, "radius": 600.0},
}

# -----------------------
# Utilitários
# -----------------------
def norm_text(s: str) -> str:
    return (s or "").strip()

def center_of_entity(e):
    try:
        if e.dxftype() == "TEXT":
            p = e.dxf.insert
            return (float(p.x), float(p.y))
        elif e.dxftype() == "MTEXT":
            p = e.dxf.insert
            return (float(p.x), float(p.y))
    except Exception:
        pass
    return (0.0, 0.0)

def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def extract_id_from_text(text, pilar_re):
    m = pilar_re.search(text)
    if m:
        return m.group().upper().replace(" ", "")
    return None

def extract_sec_from_text(text, sec_re):
    m = sec_re.search(text)
    if m:
        return re.sub(r"\s+", "", m.group()).upper()
    return None

def read_text_entities(msp, layer_whitelist=None, layer_blacklist=None):
    out = []
    for e in msp.query("TEXT MTEXT"):
        layer = e.dxf.layer if e.dxf.hasattr("layer") else ""
        if layer_whitelist and layer not in layer_whitelist:
            continue
        if layer_blacklist and layer in layer_blacklist:
            continue
        if e.dxftype() == "TEXT":
            txt = e.dxf.text
        else:
            try:
                txt = e.plain_text()
            except Exception:
                txt = e.text
        txt = norm_text(txt)
        if not txt:
            continue
        pos = center_of_entity(e)
        out.append({"type": e.dxftype(), "layer": layer, "text": txt, "pos": pos})
    return out

def read_block_inserts(msp, block_name_filter=None, layer_whitelist=None, layer_blacklist=None):
    out = []
    for ins in msp.query("INSERT"):
        layer = ins.dxf.layer if ins.dxf.hasattr("layer") else ""
        if layer_whitelist and layer not in layer_whitelist:
            continue
        if layer_blacklist and layer in layer_blacklist:
            continue
        blkname = ins.dxf.name if ins.dxf.hasattr("name") else ""
        if block_name_filter and not re.search(block_name_filter, blkname, re.IGNORECASE):
            continue
        p = ins.dxf.insert
        pos = (float(p.x), float(p.y))
        attrs = {}
        try:
            for att in ins.attribs:
                tag = norm_text(att.dxf.tag).upper()
                val = norm_text(att.dxf.text)
                attrs[tag] = val
        except Exception:
            pass
        out.append({"name": blkname, "layer": layer, "pos": pos, "attrs": attrs})
    return out

def from_block_attrs_to_pillar(block, pilar_re, sec_re, id_candidates, sec_candidates):
    attrs = block["attrs"]
    id_val = None
    sec_val = None
    for t in id_candidates:
        if t in attrs and pilar_re.search(attrs[t]):
            id_val = extract_id_from_text(attrs[t], pilar_re)
            break
    for t in sec_candidates:
        if t in attrs and sec_re.search(attrs[t]):
            sec_val = extract_sec_from_text(attrs[t], sec_re)
            break
    if not id_val:
        for v in attrs.values():
            if pilar_re.search(v):
                id_val = extract_id_from_text(v, pilar_re)
                break
    if not sec_val:
        for v in attrs.values():
            if sec_re.search(v):
                sec_val = extract_sec_from_text(v, sec_re)
                break
    if id_val:
        return {"id": id_val, "section": sec_val, "source": "block", "pos": block["pos"]}
    return None

def associate_text_pairs(text_entities, pilar_re, sec_re, radius):
    ids = []
    secs = []
    for t in text_entities:
        tid = extract_id_from_text(t["text"], pilar_re)
        if tid:
            ids.append({"id": tid, "pos": t["pos"], "layer": t["layer"], "raw": t["text"]})
            continue
        s = extract_sec_from_text(t["text"], sec_re)
        if s:
            secs.append({"section": s, "pos": t["pos"], "layer": t["layer"], "raw": t["text"]})
    pairs = []
    r2 = radius * radius
    for item in ids:
        best = None
        best_d = 1e30
        for s in secs:
            d2 = dist2(item["pos"], s["pos"])
            if d2 <= r2 and d2 < best_d:
                best_d = d2
                best = s
        pairs.append({
            "id": item["id"],
            "section": best["section"] if best else None,
            "source": "text-nearest",
            "pos": item["pos"]
        })
    return pairs

def dedupe_keep_best(items):
    best = {}
    prio = {"block": 2, "text-nearest": 1, "other": 0}
    for it in items:
        key = it["id"]
        score = prio.get(it.get("source", "other"), 0) + (1 if it.get("section") else 0)
        if key not in best:
            best[key] = (score, it)
        else:
            if score > best[key][0]:
                best[key] = (score, it)
    return [v[1] for v in best.values()]

def normalize_section(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.replace("x","/").replace("X","/").replace("×","/")
    s = re.sub(r"\s+", "", s)
    return s

# -----------------------
# Extração a partir de bytes DXF (usa ezdxf via temp file)
# -----------------------
def extract_from_dxf_bytes(dxf_bytes, pilar_re, sec_re, radius, block_name_filter=None,
                           layer_whitelist=None, layer_blacklist=None,
                           id_candidates=None, sec_candidates=None):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dxf")
    tmp.write(dxf_bytes); tmp.flush(); tmp.close()
    try:
        doc = ezdxf.readfile(tmp.name)
        msp = doc.modelspace()
    except Exception as e:
        try: os.unlink(tmp.name)
        except: pass
        raise e

    try:
        texts = read_text_entities(msp, layer_whitelist=layer_whitelist, layer_blacklist=layer_blacklist)
        inserts = read_block_inserts(msp, block_name_filter=block_name_filter or None,
                                     layer_whitelist=layer_whitelist, layer_blacklist=layer_blacklist)
        pillars_from_blocks = []
        for b in inserts:
            hit = from_block_attrs_to_pillar(b, pilar_re, sec_re, id_candidates or set(), sec_candidates or set())
            if hit:
                pillars_from_blocks.append(hit)
        pillars_from_text = associate_text_pairs(texts, pilar_re, sec_re, radius)
        merged = dedupe_keep_best(pillars_from_blocks + pillars_from_text)
        if merged:
            df = pd.DataFrame(merged)
        else:
            df = pd.DataFrame(columns=["id","section","source","pos"])
        return df, texts
    finally:
        try: os.unlink(tmp.name)
        except: pass

# -----------------------
# Expansão de IDs equivalentes (P1=P2=P3 -> linhas separadas P1, P2, P3)
# -----------------------
def expand_equivalent_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(columns=["id","section","source","pos"])
    expanded = []
    for _, row in df.iterrows():
        raw_id = str(row.get("id",""))
        if "=" in raw_id:
            parts = [p.strip().upper() for p in raw_id.split("=") if p.strip()]
        else:
            parts = [raw_id.strip().upper()] if raw_id else []
        for part in parts:
            new_row = row.copy()
            new_row["id"] = part
            expanded.append(new_row)
    if expanded:
        out = pd.DataFrame(expanded)
        for c in ["id","section","source","pos"]:
            if c not in out.columns:
                out[c] = None
        return out.reset_index(drop=True)
    else:
        return pd.DataFrame(columns=["id","section","source","pos"])

# -----------------------
# Comparação simples (DXF base x outro) - utilitário
# -----------------------
def compare_two_sets(base_df: pd.DataFrame, other_df: pd.DataFrame, other_name: str) -> pd.DataFrame:
    base = base_df.copy()
    other = other_df.copy()
    base["section_norm"]  = base["section"].fillna("").map(normalize_section)
    other["section_norm"] = other["section"].fillna("").map(normalize_section)
    base_small  = base[["id","section_norm"]].drop_duplicates().rename(columns={"section_norm":"section_base"})
    other_small = other[["id","section_norm"]].drop_duplicates().rename(columns={"section_norm":"section_comp"})

    merged = base_small.merge(other_small, on="id", how="outer")
    def status_row(r):
        sb, sc = r.get("section_base",""), r.get("section_comp","")
        if pd.notna(sb) and pd.notna(sc):
            if sb == sc:
                return "OK"
            else:
                return "Seção diferente"
        elif pd.notna(sb) and (pd.isna(sc) or sc == ""):
            return "Falta no outro"
        elif pd.notna(sc) and (pd.isna(sb) or sb == ""):
            return "Extra no outro"
        return "Indefinido"
    merged["status"] = merged.apply(status_row, axis=1)
    merged.insert(0, "Arquivo Comparado", other_name)
    return merged[["Arquivo Comparado","id","section_base","section_comp","status"]]

# -----------------------
# Tabela consolidada por ID (colunas para cada arquivo comparado + Situação)
# -----------------------
def build_nomenclature_consolidated_table(base_df: pd.DataFrame, other_dfs: dict) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return pd.DataFrame(columns=["ID","Ocorrências DXF Base", *(other_dfs.keys()), "Situação"])

    base_ids = sorted(base_df["id"].dropna().astype(str).unique())
    base_counts = base_df["id"].value_counts()

    rows = []
    other_names = list(other_dfs.keys())
    for pid in base_ids:
        row = {"ID": pid, "Ocorrências DXF Base": int(base_counts.get(pid, 0))}
        present_counter = 0
        for name in other_names:
            df = other_dfs.get(name, pd.DataFrame())
            count = int((df["id"].astype(str) == pid).sum()) if not df.empty else 0
            row[name] = count
            if count > 0:
                present_counter += 1
        if present_counter == 0:
            row["Situação"] = "❌ Não encontrado"
        elif present_counter == 1:
            row["Situação"] = "✅ Ok"
        else:
            row["Situação"] = "⚠️ Extra"
        rows.append(row)

    cols = ["ID", "Ocorrências DXF Base"] + other_names + ["Situação"]
    df_out = pd.DataFrame(rows)
    existing_cols = [c for c in cols if c in df_out.columns]
    return df_out[existing_cols]

# -----------------------
# LDF parser (pilares e vigas)  **CORRIGIDO: regex de cabeçalho**
# -----------------------
LDF_HEADER_VIGAS   = re.compile(r"^\$\s+Dimensões de vigas", re.IGNORECASE)
LDF_HEADER_PILARES = re.compile(r"^\$\s+Dimensões de pilares", re.IGNORECASE)
LDF_SECTION_RE     = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*[/xX×]\s*([0-9]+(?:[.,][0-9]+)?)")
LDF_ID_VIGA        = re.compile(r"^[Vv][A-Za-z0-9]+$")
LDF_ID_PILAR       = re.compile(r"^[Pp][A-Za-z0-9]+$")

def _norm_num(tok: str) -> str:
    tok = tok.replace(",", ".")
    try:
        v = float(tok)
        if abs(v - int(v)) < 1e-9:
            return str(int(v))
        return f"{v:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return tok

def parse_ldf_bytes(ldf_bytes: bytes):
    """Retorna dois DataFrames: (df_vigas, df_pilares) com colunas: id, section."""
    text = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            text = ldf_bytes.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        return pd.DataFrame(columns=["id","section"]), pd.DataFrame(columns=["id","section"])

    lines = text.splitlines()
    current = None
    vigas = []
    pilares = []

    for raw in lines:
        line = raw.rstrip()

        if LDF_HEADER_VIGAS.match(line.strip()):
            current = "vigas"; continue
        if LDF_HEADER_PILARES.match(line.strip()):
            current = "pilares"; continue
        if current not in ("vigas", "pilares"):
            continue

        msec = LDF_SECTION_RE.search(line)
        if not msec:
            continue

        stripped = line.strip()
        if not stripped:
            continue
        ident = stripped.split()[0]

        if current == "vigas" and not LDF_ID_VIGA.match(ident):
            continue
        if current == "pilares" and not LDF_ID_PILAR.match(ident):
            continue

        a = _norm_num(msec.group(1))
        b = _norm_num(msec.group(2))
        section = f"{a}/{b}"
        rec = {"id": ident.upper(), "section": section}
        if current == "vigas":
            vigas.append(rec)
        else:
            pilares.append(rec)

    df_v = pd.DataFrame(vigas) if vigas else pd.DataFrame(columns=["id","section"])
    df_p = pd.DataFrame(pilares) if pilares else pd.DataFrame(columns=["id","section"])
    return df_v, df_p

# -----------------------
# UI (layout principal)
# -----------------------
st.markdown('<h1 class="main-title">🔧 DXF Controle de Peças</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ Configurações</div>', unsafe_allow_html=True)

    # Perfis
    profile_names = list(DEFAULT_PROFILES.keys())
    profile_selected = st.selectbox("🎯 Carregar perfil", ["-- selecionar --"] + profile_names)

    if profile_selected and profile_selected != "-- selecionar --":
        initial_pilar = DEFAULT_PROFILES[profile_selected]["pilar"]
        initial_sec = DEFAULT_PROFILES[profile_selected]["secao"]
        initial_radius = DEFAULT_PROFILES[profile_selected]["radius"]
    else:
        initial_pilar = DEFAULT_PILAR_REGEX
        initial_sec = DEFAULT_SECAO_REGEX
        initial_radius = 400.0

    st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)
    pilar_pattern = st.text_input("🏗️ Regex - Nomenclatura do pilar", value=initial_pilar,
                                 help="Ex.: PT1A, P1, PILAR 1. Suporta P1=P2=P3")
    sec_pattern = st.text_input("📐 Regex - Seção", value=initial_sec, help="Ex.: 19x199, 19/19")
    radius = st.number_input("📏 Raio associação ID ↔ Seção (unidades DXF)",
                             value=float(initial_radius), min_value=1.0,
                             help="Distância máxima para associar ID e seção")

    st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

    with st.expander("🔍 Filtros Avançados", expanded=False):
        block_name_filter = st.text_input("Filtrar blocos (regex, opcional)", value="")
        layers_csv = st.text_input("Layers (whitelist, separado por vírgula)", value="")
        exclude_layers_csv = st.text_input("Ignorar layers (separado por vírgula)", value="")

    with st.expander("🏷️ Configuração de Tags", expanded=False):
        st.write("Tags comuns em blocos (ID e Seção)")
        id_attr_tags = st.text_input("Possíveis tags de ID",
                                     value="ID,NOME,NAME,PILAR,PILAR_ID")
        sec_attr_tags = st.text_input("Possíveis tags de seção",
                                      value="SECAO,SEC,SECTION,DIM,DIMENSOES")

    st.markdown('</div>', unsafe_allow_html=True)

    # Upload LDF (novo)
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📄 Upload LDF (opcional)</div>', unsafe_allow_html=True)
    uploaded_ldf = st.file_uploader("📎 Envie o arquivo .LDF", type=["ldf"],
                                    help="O LDF contém, entre outros dados, as seções de vigas e pilares.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🗂️ Outros DXFs para comparar (opcional)</div>', unsafe_allow_html=True)
    uploaded_others = st.file_uploader("📎 Envie 1 ou mais DXFs para comparar", type=["dxf"],
                                       accept_multiple_files=True,
                                       help="Arquivos que serão comparados contra o DXF Base.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📁 DXF Base (principal)</div>', unsafe_allow_html=True)
    uploaded_base = st.file_uploader("📎 Envie o DXF Base (.dxf)", type=["dxf"], help="Este é o arquivo principal (referência)")
    run_btn = st.button("🚀 Processar", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Pré-processamento de regex e inputs
# -----------------------
try:
    pilar_re = re.compile(pilar_pattern, re.IGNORECASE)
except Exception:
    st.error("⚠️ Regex de pilar inválida. Corrija antes de processar.")
    pilar_re = None

try:
    sec_re = re.compile(sec_pattern, re.IGNORECASE)
except Exception:
    st.error("⚠️ Regex de seção inválida. Corrija antes de processar.")
    sec_re = None

layer_whitelist = set([s.strip() for s in layers_csv.split(",") if s.strip()]) if layers_csv else None
layer_blacklist = set([s.strip() for s in exclude_layers_csv.split(",") if s.strip()]) if exclude_layers_csv else None
id_candidates = set([s.strip().upper() for s in id_attr_tags.split(",") if s.strip()])
sec_candidates = set([s.strip().upper() for s in sec_attr_tags.split(",") if s.strip()])

# -----------------------
# Processamento principal
# -----------------------
if run_btn:
    if not uploaded_base:
        st.error("⚠️ Envie o DXF Base para iniciar.")
    elif not (pilar_re and sec_re):
        st.error("⚠️ Corrija as regex antes de processar.")
    else:
        # Processa DXF base
        with st.spinner("🔄 Lendo DXF Base..."):
            base_bytes = uploaded_base.read()
            try:
                base_df, base_texts = extract_from_dxf_bytes(
                    base_bytes, pilar_re, sec_re, radius,
                    block_name_filter if block_name_filter else None,
                    layer_whitelist, layer_blacklist,
                    id_candidates, sec_candidates
                )
            except Exception as e:
                st.error(f"❌ Erro ao abrir DXF Base: {e}")
                base_df = pd.DataFrame(columns=["id","section","source","pos"])
                base_texts = []

        # Expande IDs equivalentes no base
        base_df = expand_equivalent_ids(base_df)

        # Exibe resumo e downloads da extração
        with st.container():
            st.success(f"✅ {len(base_df)} elementos identificados no DXF Base (após expandir equivalências).")
            st.markdown('<div class="section-header">📊 Elementos do DXF Base</div>', unsafe_allow_html=True)
            if not base_df.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total de Peças", len(base_df))
                c2.metric("Com Seção", len(base_df[base_df['section'].notna() & (base_df['section']!="")]))
                c3.metric("Apenas ID", len(base_df[base_df['section'].isna() | (base_df['section']=="")]))
                st.dataframe(base_df, use_container_width=True)

                csv_bytes = base_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
                json_bytes = base_df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")
                cold1, cold2 = st.columns(2)
                cold1.download_button("📄 Baixar CSV (Extração Base)", data=csv_bytes, file_name="base_extracao.csv", mime="text/csv")
                cold2.download_button("📋 Baixar JSON (Extração Base)", data=json_bytes, file_name="base_extracao.json", mime="application/json")
            else:
                st.warning("⚠️ Nenhum elemento encontrado no DXF Base com os critérios atuais.")

        # -----------------------
        # Comparação DXF Base × LDF (forçada conforme perfil)
        # -----------------------
        if uploaded_ldf:
            ldf_bytes = uploaded_ldf.read()
            ldf_vigas, ldf_pilares = parse_ldf_bytes(ldf_bytes)

            # Decide o modo de comparação (força por perfil)
            force_tag = None
            if profile_selected and profile_selected != "-- selecionar --":
                low = profile_selected.lower()
                if "pilar" in low:
                    force_tag = "Pilares"
                elif "viga" in low:
                    force_tag = "Vigas"

            def _compare_and_show(tag, df_ldf):
                temp_ldf = df_ldf.copy()
                temp_ldf["section"] = temp_ldf["section"].fillna("").map(normalize_section)

                temp_base = base_df.copy()
                temp_base["section_norm"] = temp_base["section"].fillna("").map(normalize_section)

                merged_cmp = temp_ldf.rename(columns={"section":"section_ldf"}).merge(
                    temp_base[["id","section_norm"]],
                    on="id", how="outer"
                )

                def _status(r):
                    ldf_s = r.get("section_ldf","")
                    dxf_s = r.get("section_norm","")
                    if ldf_s and dxf_s:
                        return "OK" if ldf_s == dxf_s else "Seção diferente"
                    elif ldf_s and not dxf_s:
                        return "Falta no DXF"
                    elif dxf_s and not ldf_s:
                        return "Extra no LDF"
                    return "Indefinido"

                if not merged_cmp.empty:
                    merged_cmp["status"] = merged_cmp.apply(_status, axis=1)
                    st.markdown(f'<div class="section-header">📊 Comparação DXF Base × LDF — {tag}</div>', unsafe_allow_html=True)
                    st.dataframe(merged_cmp[["id","section_ldf","section_norm","status"]], use_container_width=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("OK", int((merged_cmp["status"]=="OK").sum()))
                    c2.metric("Diferenças", int((merged_cmp["status"]=="Seção diferente").sum()))
                    c3.metric("Faltando no DXF", int((merged_cmp["status"]=="Falta no DXF").sum()))
                    c4.metric("Extras no LDF", int((merged_cmp["status"]=="Extra no LDF").sum()))
                    st.download_button(
                        f"📥 Baixar Comparação DXF×LDF ({tag})",
                        data=merged_cmp.to_csv(index=False, encoding="utf-8-sig"),
                        file_name=f"comparacao_dxf_base_ldf_{tag.lower()}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"⚠️ LDF ({tag}) processado, mas sem linhas válidas para comparar.")

            if force_tag == "Pilares":
                _compare_and_show("Pilares", ldf_pilares)
            elif force_tag == "Vigas":
                _compare_and_show("Vigas", ldf_vigas)
            else:
                tab1, tab2 = st.tabs(["Vigas (LDF)", "Pilares (LDF)"])
                with tab1:
                    _compare_and_show("Vigas", ldf_vigas)
                with tab2:
                    _compare_and_show("Pilares", ldf_pilares)

        # Comparação DXF Base x Outros DXFs (múltiplos)
        other_dfs = {}
        consolidated_list = []
        if uploaded_others:
            with st.spinner(f"🔍 Comparando {len(uploaded_others)} DXF(s) com o DXF Base..."):
                for f in uploaded_others:
                    try:
                        other_bytes = f.read()
                        other_df, other_texts = extract_from_dxf_bytes(
                            other_bytes, pilar_re, sec_re, radius,
                            block_name_filter if block_name_filter else None,
                            layer_whitelist, layer_blacklist,
                            id_candidates, sec_candidates
                        )
                        other_df = expand_equivalent_ids(other_df)
                        other_dfs[f.name] = other_df
                        cmp_df = compare_two_sets(base_df, other_df, f.name)
                        consolidated_list.append(cmp_df)
                    except Exception as e:
                        st.error(f"Erro ao processar {f.name}: {e}")

        if consolidated_list:
            consolidated = pd.concat(consolidated_list, ignore_index=True)
            with st.container():
                st.markdown('<div class="section-header">🧾 Comparação detalhada (Base × cada arquivo)</div>', unsafe_allow_html=True)
                st.dataframe(consolidated, use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("OK", int((consolidated["status"]=="OK").sum()))
                c2.metric("Seção diferente", int((consolidated["status"]=="Seção diferente").sum()))
                c3.metric("Falta no outro", int((consolidated["status"]=="Falta no outro").sum()))
                c4.metric("Extra no outro", int((consolidated["status"]=="Extra no outro").sum()))
                st.download_button("📥 Baixar Comparação detalhada (CSV)",
                                   data=consolidated.to_csv(index=False, encoding="utf-8-sig"),
                                   file_name="comparacao_detalhada.csv", mime="text/csv")

        # Tabela consolidada por ID (por arquivos comparados) e Situação
        if other_dfs:
            with st.container():
                st.markdown('<div class="section-header">🧮 Comparação DXF Base × Outros DXFs (Consolidado por ID)</div>', unsafe_allow_html=True)
                nom_df = build_nomenclature_consolidated_table(base_df, other_dfs)
                st.dataframe(nom_df, use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Ok", int((nom_df["Situação"]=="✅ Ok").sum()))
                c2.metric("⚠️ Extra", int((nom_df["Situação"]=="⚠️ Extra").sum()))
                c3.metric("❌ Não encontrado", int((nom_df["Situação"]=="❌ Não encontrado").sum()))
                st.download_button("📥 Baixar Comparação DXF×DXFs (CSV)",
                                   data=nom_df.to_csv(index=False, encoding="utf-8-sig"),
                                   file_name="comparacao_dxf_base_outros_dxfs_consolidado.csv",
                                   mime="text/csv")

        # Gráfico de localização (DXF Base)
        with st.container():
            st.markdown('<div class="section-header">📍 Localização dos elementos no DXF Base</div>', unsafe_allow_html=True)
            if not base_df.empty and "pos" in base_df.columns:
                x_vals = [p[0] for p in base_df["pos"]]
                y_vals = [p[1] for p in base_df["pos"]]
                ids = base_df["id"].tolist()

                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(x_vals, y_vals, s=80, edgecolors="white", linewidths=0.8, color='#FF9500')
                for i, txt in enumerate(ids):
                    ax.annotate(txt, (x_vals[i], y_vals[i]), textcoords="offset points", xytext=(5, 5), fontsize=9, color="white")
                ax.set_facecolor("#0d1117")
                fig.patch.set_facecolor("#0d1117")
                ax.tick_params(colors="white")
                ax.set_xlabel("Coordenada X", color="white")
                ax.set_ylabel("Coordenada Y", color="white")
                ax.set_title("Mapa dos elementos detectados no DXF Base", color="white")
                ax.grid(True, color="#333333")
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.info("Nenhuma posição encontrada para exibir no gráfico.")

# Mensagem inicial
if not run_btn:
    st.info("📤 Envie o **DXF Base** (obrigatório). Opcionalmente, envie um **LDF** e/ou **outros DXFs** e clique em **Processar**.")
