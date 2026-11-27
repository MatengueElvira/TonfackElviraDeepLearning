"""Application web - Module 1 : detection de pneumonie sur radiographie thoracique.

"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from matplotlib import colormaps
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

st.set_page_config(page_title="Radiographie thoracique - Pneumonie", layout="wide")

WEIGHTS = HERE / "pneumonia_best.pt"
CONFIG = HERE / "model_config.json"


@st.cache_resource(show_spinner="Chargement du modele...")
def get_model():
    import pneumonia_model as pm
    model, cfg = pm.load_for_inference(WEIGHTS, CONFIG, device="cpu")
    return pm, model, cfg


def overlay(gray_u8, cam, alpha=0.4):
    """Superpose la carte Grad-CAM (valeurs dans [0,1]) a l'image grise."""
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    base = np.stack([gray_u8] * 3, axis=-1).astype(np.float32)
    out = (1 - alpha) * base + alpha * heat.astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def page_accueil():
    st.title("Projet Deep Learning - M2 Data Science")
    st.write(
        "Cette application regroupe deux modules. Le module 1 classe des radiographies "
        "thoraciques en NORMAL ou PNEUMONIA. Le module 2 prevoit le cours de l'action "
        "Samsung sur 20 jours ouvres avec Neural Prophet."
    )
    st.warning("Outil pedagogique. Il ne remplace pas l'avis d'un medecin et ne doit pas servir au diagnostic.")


def page_module1():
    st.title("Module 1 - Detection de pneumonie")
    if not (WEIGHTS.exists() and CONFIG.exists() and (HERE / "pneumonia_model.py").exists()):
        st.error(
            "Fichiers du modele introuvables. Copie pneumonia_best.pt, model_config.json et "
            "pneumonia_model.py (produits par le notebook) a cote de app.py."
        )
        return
    pm, model, cfg = get_model()

    with st.sidebar:
        st.subheader("Reglages")
        use_clinical = st.checkbox("Utiliser le seuil clinique", value=True,
                                   help="Seuil choisi sur la validation pour garantir un rappel eleve.")
        custom = st.slider("Seuil manuel", 0.01, 0.99, 0.50, 0.01, disabled=use_clinical)
        show_cam = st.checkbox("Afficher Grad-CAM", value=True)
    threshold = float(cfg["threshold"]) if use_clinical else float(custom)

    file = st.file_uploader("Televerse une radiographie (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if file is None:
        st.info("Aucune image pour le moment.")
    else:
        try:
            img = Image.open(file)
            img.load()
        except Exception:
            st.error("Image illisible. Essaie un autre fichier jpg ou png.")
            return
        cfg_run = {**cfg, "threshold": threshold}
        with st.spinner("Analyse en cours..."):
            res = pm.predict_pil(model, cfg_run, img, with_cam=show_cam)
        p = res["prob_pneumonia"]

        c1, c2 = st.columns(2)
        with c1:
            st.image(res["image_gray"], caption="Image envoyee au modele", use_container_width=True)
        with c2:
            if show_cam and res["cam"] is not None:
                st.image(overlay(res["image_gray"], res["cam"]),
                         caption="Grad-CAM : zones qui poussent vers la classe pneumonie", use_container_width=True)
        if res["label"] == "PNEUMONIA":
            st.error(f"Resultat : PNEUMONIA (probabilite {p:.1%}, seuil {threshold:.3f})")
        else:
            st.success(f"Resultat : NORMAL (probabilite de pneumonie {p:.1%}, seuil {threshold:.3f})")
        st.progress(min(max(p, 0.0), 1.0), text=f"Probabilite de pneumonie : {p:.1%}")
        st.caption("Une carte concentree sur les poumons est rassurante ; une carte sur les bords ou du texte "
                   "signale un possible raccourci appris par le modele.")

    with st.expander("A propos du modele"):
        st.write(f"Architecture : {cfg['arch']}, image {cfg['img_size']} * {cfg['img_size']} pixels, "
                 f"{cfg.get('params', 0) / 1e6:.2f} millions de parametres.")
        st.write(f"Seuil clinique : {cfg['threshold']:.3f} (rappel vise sur la validation : {cfg.get('target_recall', 0):.0%}).")
        for key, title in [("val_metrics", "Validation"), ("test_metrics_clinical_threshold", "Test, seuil clinique"),
                           ("test_metrics_threshold_0_5", "Test, seuil 0.5")]:
            if key in cfg:
                st.write(title)
                st.dataframe({k: [round(float(v), 4)] for k, v in cfg[key].items()}, hide_index=True)
        st.caption(cfg.get("disclaimer", ""))


def page_module2():
    import pandas as pd
    import matplotlib.pyplot as plt
    st.title("Module 2 - Prevision du cours Samsung sur 20 jours")
    files = ["samsung_history.csv", "samsung_forecast.csv", "samsung_backtest.csv", "samsung_metrics.json"]
    if not all((HERE / f).exists() for f in files):
        st.error("Fichiers introuvables. Copie " + ", ".join(files) + " (produits par le notebook Samsung_NeuralProphet) a cote de app.py.")
        return
    hist = pd.read_csv(HERE / "samsung_history.csv", parse_dates=["ds"])
    fc = pd.read_csv(HERE / "samsung_forecast.csv", parse_dates=["ds"])
    bt = pd.read_csv(HERE / "samsung_backtest.csv", parse_dates=["ds"])
    info = json.loads((HERE / "samsung_metrics.json").read_text())
    if info.get("quick_run"):
        st.warning("Ces fichiers viennent d'un test rapide : les resultats n'ont aucune valeur.")

    n_hist = st.slider("Jours d'historique affiches", 30, min(len(hist), 500), 120, 10)
    fig, ax = plt.subplots(figsize=(10, 4))
    h = hist.iloc[-n_hist:]
    ax.plot(h.ds, h.y, label="historique", color="#4C78A8")
    ax.plot(fc.ds, fc.yhat, label="prevision", color="#E45756")
    ax.fill_between(fc.ds, fc.lo, fc.hi, alpha=0.2, color="#E45756", label="intervalle 90 %")
    ax.set_ylabel("cours (wons)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dernier cours connu", f"{info['dernier_cours']:,.0f}", help=info["derniere_date_connue"])
    c2.metric("Prevision a 20 jours", f"{fc.yhat.iloc[-1]:,.0f}", f"{fc.yhat.iloc[-1] / info['dernier_cours'] - 1:+.1%}")
    c3.metric("Intervalle 90 % a 20 jours", f"{fc.lo.iloc[-1]:,.0f} - {fc.hi.iloc[-1]:,.0f}")
    st.dataframe(fc.assign(ds=fc.ds.dt.date).round(0), hide_index=True)
    st.download_button("Telecharger la prevision (CSV)", fc.to_csv(index=False), "samsung_forecast.csv", "text/csv")

    with st.expander("Fiabilite : test sur les 20 derniers jours connus"):
        fig2, ax2 = plt.subplots(figsize=(10, 3.6))
        ax2.plot(bt.ds, bt.reel, label="reel", color="k")
        ax2.plot(bt.ds, bt.prevision, label="Neural Prophet", color="#E45756")
        ax2.plot(bt.ds, bt.naif, "--", label="modele naif", color="grey")
        ax2.fill_between(bt.ds, bt.lo, bt.hi, alpha=0.2, color="#E45756")
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        t, n = info["test"], info["test_naif"]
        st.write(f"Erreur moyenne (MAE) : {t['mae']:,.0f} wons pour Neural Prophet contre {n['mae']:,.0f} pour le modele naif "
                 f"(rapport {t['rapport_au_naif']:.2f}, inferieur a 1 = meilleur que le naif).")
        st.write(f"Configuration retenue : {info['configuration']}. Couverture de l'intervalle sur le test : "
                 f"{info['couverture_intervalle_test']:.0%}.")
    st.caption("Les cours boursiers sont tres difficiles a predire : le modele naif est souvent aussi bon. " + info.get("avertissement", ""))


PAGES = {"Accueil": page_accueil, "Module 1 - Pneumonie": page_module1, "Module 2 - Samsung": page_module2}
with st.sidebar:
    choice = st.radio("Navigation", list(PAGES))
PAGES[choice]()
