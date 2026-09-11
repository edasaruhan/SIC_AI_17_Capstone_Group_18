import sys
import base64
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import altair as alt

from src.pipeline import run_pipeline
from src.brand_alignment import (load_brand_data, validate_alignment)
from src.ctr_model import MAIN_CONTENT_MODEL
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="OnBrand AdCopy",
    layout="wide",
    initial_sidebar_state="collapsed",
)

css = (APP_DIR / "theme.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Marka renkleri (tone_category türevi)
BRAND_COLORS = {
    "samsung": "#7FB3E8",
    "duolingo": "#5FB89C",
    "nike": "#D96C4F",
}

# Üslüp rozet sınıfları
STYLE_CLASS = {
    "emotional": "duygusal",
    "informative": "bilgilendirici",
    "urgency": "aciliyet",
    "humorous": "mizahi",
}

STYLE_LABEL = {
    "emotional": "Duygusal",
    "informative": "Bilgilendirici",
    "urgency": "Aciliyet vurgulu",
    "humorous": "Mizahi",
}

SCORE_HINTS = {
    "ctr": "Bu metnin geçmiş verilere göre tahmini etkileşim potansiyeli. Yüksek = daha çok beğeni/paylaşım/yorum beklenir.",
    "brand": "Bu metnin markanın onaylı referans metinlerine anlamsal yakınlığı. Yüksek = marka sesine daha sadık.",
    "composite": "Yukarıdaki iki skorun, seçtiğiniz ağırlığa göre birleşimi. Sıralama bu skora göre yapılır.",
}


def inject_topbar():
    logo = base64.b64encode((APP_DIR / "assets" / "sic.jpg").read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .topbar .sic-logo {{
            height: 52px;
            width: auto;
            max-width: 220px;
            object-fit: contain;
            display: block;
            border-radius: 6px;
            background: #000;
            padding: 6px 10px;
        }}
        </style>
        <div class="topbar">
          <img class="sic-logo" src="data:image/jpeg;base64,{logo}" alt="SIC">
          <div class="title-block">
            <h1>OnBrand AdCopy</h1>
            <div class="subtitle">Marka sesine sadık, performansı ölçülmüş reklam metni üretici</div>
          </div>
          <div class="badge">Marka Sesi + Performans</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_bar(label, value, css_class, hint, fmt="{:.1f}"):
    return (
        f'<div class="score-cell">'
        f'<span class="score-label {css_class}">{label}</span>'
        f'<span class="score-value {css_class}">{fmt.format(value)}</span>'
        f'<span class="score-hint">{hint}</span>'
        f'</div>'
    )


def comp_bar(score):
    pct = max(0.0, min(100.0, score))
    return (
        f'<div class="comp-bar-track">'
        f'<div class="comp-bar-fill" style="width:{pct:.1f}%"></div>'
        f'</div>'
    )


def render_result_card(index, v, weight_ctr=0.6):
    corr = v.get("correction_info", {})
    style_cls = STYLE_CLASS.get(v.get("style", ""), "")
    style_lbl = STYLE_LABEL.get(v.get("style", ""), v.get("style", ""))

    badges = ""
    if v.get("needs_human_review"):
        badges += '<span class="human-badge">İnsan onayı gerekli</span>'
    elif corr.get("corrected"):
        badges += '<span class="corrected-badge">↺ Marka tonuna göre yeniden yazıldı</span>'

    # 10/10 Mutlak skor gösterimi: karttaki skorlar altındaki açıklamayla BİREBİR tutar.
    disp_ctr = v["ctr_benchmark"]
    disp_brand = round(float(v["brand_score"]), 1)
    # Bileşik = kullanıcının seçtiği ağırlıkla bu iki mutlak skorun ortalaması (0.6 vs 0.4 gibi sabit DEĞİL)
    disp_comp = round(weight_ctr * disp_ctr + (1 - weight_ctr) * disp_brand, 1)

    sb_ctr = score_bar("CTR Skoru", disp_ctr, "ctr", SCORE_HINTS["ctr"])
    sb_brand = score_bar("Marka Uyumu", disp_brand, "brand", SCORE_HINTS["brand"])
    sb_comp = score_bar("Bileşik Skor", disp_comp, "comp", SCORE_HINTS["composite"])
    cb = comp_bar(disp_comp)

    # Gri açıklama metni sadeleştirildi: tek gerçek ham kaynak değerler
    raw_hint = (
        f'<div class="raw-scores-hint">'
        f'Ham Değerler: Tahmini CTR %{v["ctr_score"]:.1f} '
        f'| SBERT Anlamsal Benzerlik: %{disp_brand:.1f}'
        f'</div>'
    )

    card = (
        f'<div class="result-card">'
        f'<div class="rank-row">'
        f'<span class="rank-badge">#{index + 1}</span>'
        f'<span class="style-badge {style_cls}">{style_lbl}</span>'
        f'{badges}'
        f'</div>'
        f'<div class="copy-text">{v["text"]}</div>'
        f'<div class="score-grid">'
        f'{sb_ctr}'
        f'{sb_brand}'
        f'{sb_comp}'
        f'{cb}'
        f'{raw_hint}'
        f'</div>'
        f'</div>'
    )
    st.markdown(card, unsafe_allow_html=True)

    if corr.get("corrected"):
        with st.expander("↺ Öncesi / sonrası karşılaştırması"):
            col_a, col_b = st.columns(2)
            col_a.markdown("**Yeniden yazılmadan önce**")
            col_a.write(corr.get("original_text", v["text"]))
            col_b.markdown("**Yeniden yazıldıktan sonra**")
            col_b.write(v["text"])
            st.caption(
                f"Marka uyumu: {corr['original_brand_score']:.1f} → "
                f"{v['brand_score']:.1f}  |  CTR farkı: {corr['ctr_delta']:+.1f}  |  "
                f"Deneme: {corr['attempts']}"
            )


def render_side_panel(results, brand_id, brand_name):
    if not results or not results.get("ranked_variants"):
        st.markdown(
            '<div class="section-title">Sonuçları burada göreceksiniz</div>'
            '<div class="section-sub">Soldaki formu doldurup '
            '<b>"Üret ve Puanla"</b> butonuna basın. Sistem 4 farklı üslupta '
            "reklam metni üretip CTR ve marka uyumu skorlayarak sıralar.</div>",
            unsafe_allow_html=True,
        )
    else:
        weight_ctr = results.get("weight_ctr", 0.6)
        st.markdown(
            f'<div class="results-header">'
            f'<div class="section-title">{brand_name} için {len(results["ranked_variants"])} varyant '
            f'(sıralandı)</div>'
            f'<div style="color:var(--ink-muted);font-size:0.78rem">Toplam süre: '
            f'{results["timings"]["total"]:.1f}s</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for i, v in enumerate(results["ranked_variants"]):
            render_result_card(i, v, weight_ctr)


@st.cache_data(show_spinner=False)
def _get_cached_validation():
    try:
        return validate_alignment(load_brand_data())
    except Exception:
        return None


def render_why_panel():
    st.markdown(
        '<div class="section-title">Neden bu sıralama?</div>'
        '<div class="section-sub">Şeffaflık ilkesi gereği her skorun arka planı açıkça '
        "gösteriliyor — bu sistem tahmin üretir, kesin sonuç garantisi vermez.</div>",
        unsafe_allow_html=True,
    )

    tab_imp, tab_val = st.tabs(["CTR Modeli Özellik Önemi", "Marka Uyum Doğrulama"])

    with tab_imp:
        imp_path = PROJECT_ROOT / "models" / "feature_importance.json"
        if imp_path.exists():
            importances = json.loads(imp_path.read_text(encoding="utf-8"))
            feature_names = {
                "toxicity_score": "Toksiklik skoru",
                "sentiment_score": "Duygu skoru",
                "urgency_count": "Aciliyet kelime sayısı",
                "char_count": "Karakter sayısı",
                "word_count": "Kelime sayısı",
                "has_question": "Soru içeriyor mu",
                "has_exclamation": "Ünlem içeriyor mu",
                "emoji_count": "Emoji sayısı",
                "hashtag_count": "Hashtag sayısı",
                "has_hashtag": "Hashtag var mı",
                "has_mention": "@bahsetme var mı",
                "has_cta": "Harekete geçirici ifade (CTA)",
                "avg_word_len": "Ortalama kelime uzunluğu",
                "word_diversity": "Kelime çeşitliliği",
                "all_caps_ratio": "BÜYÜK harf oranı",
                "numeric_ratio": "Sayı içeren kelime oranı",
            }
            # Sadece pozitif öneme sahip özellikleri sırala
            active_items = [(feature_names.get(k, k), v) for k, v in importances.items() if v > 0]
            imp_df = pd.DataFrame(
                active_items,
                columns=["Özellik", "Önem"],
            ).sort_values("Önem", ascending=True)

            # Çelişki 3: Altair ile ekseni min değerden başlat (zero=False) —
            # barlar birbirinden ayırt edilir, tam Türkçe etiketler kesilmez.
            chart = (
                alt.Chart(imp_df)
                .mark_bar(color="#7FB3E8")
                .encode(
                    x=alt.X("Önem:Q", title="Önem", scale=alt.Scale(zero=False)),
                    y=alt.Y(
                        "Özellik:N",
                        title="",
                        sort="-x",
                        axis=alt.Axis(labelLimit=400, titlePadding=16, labelPadding=12),
                    ),
                )
                .properties(width="container", height=max(260, len(imp_df) * 32))
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "Yeni reklam metinlerini puanlayan içerik modelinin özellik önem "
                "sıralaması. Model, metin içi NLP özellikleri ve duygu/toksiklik skorlarını "
                "değerlendirerek etkileşim potansiyeli tahmin eder."
            )
        else:
            st.caption("Önce CTR modelini eğitin (models/feature_importance.json oluşur).")

    with tab_val:
        brand_data = load_brand_data()
        val = _get_cached_validation()
        if val:
            rows = [
                {
                    "Marka": brand_data[b]["brand_name"],
                    "Kendi referansı ort. skor": round(v["own_mean"], 1),
                    "Farklı marka ort. skor": round(v["cross_mean"], 1),
                    "Kendi referansı daha yüksek": "Evet ✅" if v["own_higher"] else "Hayır ❌",
                }
                for b, v in val.items()
                if b in brand_data
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(
                "Her markanın onaylı referans metinleri kendi markasına karşı, rakip "
                "markalara karşı olduğundan belirgin şekilde daha yüksek skor veriyor — "
                "skorun güvenilir bir 'marka yakınlığı' ölçüsü olduğunu doğrular."
            )
        else:
            st.caption("Marka doğrulama verisi hesaplanamadı.")

    st.markdown(
        '<div class="approach-note">'
        "<b>Bu skorlar, 16 metin-içi NLP özelliği (duygu, aciliyet, biçimlendirme) üzerinde "
        "eğitilmiş XGBoost İçerik Modeli ve çok dilli SBERT anlamsal yakınlık analizinden "
        "(paraphrase-multilingual-MiniLM-L12-v2) elde edilir — reklam metninin bağımsız "
        "etkileşim potansiyelini ve marka tonuna sadakatini ölçer.</b>"
        "</div>",
        unsafe_allow_html=True,
    )


def main():
    inject_topbar()

    brand_data = load_brand_data()
    brand_ids = list(brand_data.keys())
    brand_names = {b: brand_data[b]["brand_name"] for b in brand_ids}

    col_left, col_right = st.columns([1, 2.2], gap="large")

    with col_left:
        st.markdown(
            '<div class="section-title">Ürün & Hedef</div>',
            unsafe_allow_html=True,
        )

        product_desc = st.text_area(
            "Ürün / Hizmet Açıklaması",
            placeholder="örn. Galaxy S24 Ultra, yapay zekâ destekli kamera",
            height=120,
        )
        st.markdown(
            '<div class="field-hint">Reklamı yapılacak ürünü birkaç cümleyle tanımlayın.</div>',
            unsafe_allow_html=True,
        )

        target_audience = st.text_input(
            "Hedef Kitle",
            placeholder="örn. teknoloji meraklısı 25-35 yaş",
        )
        st.markdown(
            '<div class="field-hint">Bu reklamın kime hitap ettiğini yazın '
            "(örn. teknoloji meraklısı 25-35 yaş).</div>",
            unsafe_allow_html=True,
        )

        sel_labels = {
            b: f"{brand_names[b]}  {BRAND_COLORS.get(b, '#888')}"
            for b in brand_ids
        }
        selected_brand = st.selectbox(
            "Marka",
            options=brand_ids,
            format_func=lambda x: brand_names[x],
        )
        brand_color = BRAND_COLORS.get(selected_brand, "#888")
        st.markdown(
            f'<div class="field-hint">Üretilecek metnin hangi markanın ses tonuna '
            f'uyması gerektiğini seçin &nbsp;<span style="color:{brand_color}">●</span> '
            f'{brand_data[selected_brand]["tone_category"]}</div>',
            unsafe_allow_html=True,
        )

        weight_ctr = st.slider(
            "Ağırlık: Performans ↔ Marka Uyumu",
            0.0, 1.0, 0.6, 0.05,
            format="%.2f",
        )
        weight_brand = 1.0 - weight_ctr
        st.markdown(
            f'<div class="field-hint">Sağa kaydırırsanız sistem tıklama potansiyeli '
            f"yüksek metinleri öne çıkarır; sola kaydırırsanız marka sesine daha sadık "
            f"metinleri öne çıkarır. <b>Şu an: CTR %{round(weight_ctr*100):.0f} / "
            f"Marka %{round(weight_brand*100):.0f}</b></div>",
            unsafe_allow_html=True,
        )

        # Görev 2: Custom Brand Wizard
        with st.expander("Yeni Marka Ekle (Sihirbaz)", expanded=False):
            new_id = st.text_input("Marka ID (kısa, İngilizce, boşluk yok)", key="new_brand_id")
            new_name = st.text_input("Marka Adı", key="new_brand_name")
            new_tone = st.text_input("Ton Kategorisi (ör. Cesur, Samimi, Premium)", key="new_brand_tone")
            new_voice = st.text_area("Marka Ses Özeti (1-2 cümle)", key="new_brand_voice", height=60)
            new_refs = st.text_area(
                "Referans Reklam Metinleri (satır satır, en az 3)",
                key="new_brand_refs", height=100,
            )
            new_preferred = st.text_input("Tercih Edilen Kelimeler (virgülle)", key="new_brand_pref")
            new_forbidden = st.text_input("Yasaklı Kelimeler (virgülle)", key="new_brand_forbid")
            new_rules = st.text_area(
                "Stil Kuralları (satır satır)",
                key="new_brand_rules", height=80,
            )

            if st.button("Markayı Kaydet", key="save_brand"):
                if not new_id.strip() or not new_name.strip():
                    st.error("Marka ID ve Adı zorunludur.")
                elif not new_refs.strip():
                    st.error("En az 3 referans reklam metni gerekir.")
                else:
                    brand_path = PROJECT_ROOT / "data" / "brand_reference.json"
                    try:
                        data = json.loads(brand_path.read_text(encoding="utf-8"))
                    except Exception:
                        data = {"brands": []}

                    # Aynı ID varsa güncelle, yoksa ekle
                    existing = next((b for b in data["brands"] if b["brand_id"] == new_id.strip()), None)
                    refs_list = [r.strip() for r in new_refs.strip().splitlines() if r.strip()]
                    rules_list = [r.strip() for r in new_rules.strip().splitlines() if r.strip()]
                    new_brand = {
                        "brand_id": new_id.strip(),
                        "brand_name": new_name.strip(),
                        "tone_category": new_tone.strip() or "Genel",
                        "brand_voice_summary": new_voice.strip(),
                        "style_rules": rules_list or ["Markanın ses tonuna sadık kal"],
                        "preferred_vocabulary": [w.strip() for w in new_preferred.split(",") if w.strip()],
                        "forbidden_vocabulary": [w.strip() for w in new_forbidden.split(",") if w.strip()],
                        "reference_ad_copies": refs_list,
                    }
                    if existing:
                        existing.update(new_brand)
                    else:
                        data["brands"].append(new_brand)

                    brand_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                    st.success(f"'{new_name.strip()}' markası kaydedildi! Sayfayı yenileyin.")
                    st.rerun()

        model_ready = MAIN_CONTENT_MODEL.exists()
        if not model_ready:
            st.warning(
                "CTR içerik modeli bulunamadı. Önce ana modeli eğitin:\n"
                "`python scripts/train_main_model.py`"
            )

        generate_pressed = st.button(
            "Üret ve Puanla",
            type="primary",
            width="stretch",
            disabled=not (product_desc and target_audience and model_ready),
        )

        results = st.session_state.get("results")

    with col_right:
        if generate_pressed:
            with st.spinner("Metinler üretiliyor ve puanlanıyor..."):
                try:
                    results = run_pipeline(
                        product_desc, target_audience, selected_brand, weight_ctr
                    )
                    st.session_state["results"] = results
                except FileNotFoundError:
                    st.error(
                        "Model dosyası bulunamadı. Önce `python scripts/train_main_model.py` "
                        "çalıştırarak ana modeli eğitin."
                    )
                except Exception as e:
                    st.error(f"Bir hata oluştu: {e}")

        render_side_panel(results or {}, selected_brand, brand_names[selected_brand])

        if results and results.get("ranked_variants"):
            export_df = pd.DataFrame(results["ranked_variants"])
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Sonuclari CSV olarak indir",
                data=csv_bytes,
                file_name="onbrand_adcopy_results.csv",
                mime="text/csv",
            )

    if results and results.get("generation_errors"):
        from src.generator import classify_api_error
        n = len(results["generation_errors"])
        ok = len(results["ranked_variants"])
        first = results["generation_errors"][0]
        reason = classify_api_error(first.split(":", 1)[-1] if ":" in first else first)
        if ok > 0:
            st.warning(
                f"⚠️ {ok}/4 üslup üretilebildi, {n} tanesi başarısız oldu.\n\n"
                f"**Sebep:** {reason}\n\n"
                f"💡 Free tier API kotası günde 20 istek/model ile sınırlıdır. "
                f"Kota gece yarısı (UTC) sıfırlanır."
            )
        else:
            st.error(
                f"❌ Hiçbir üslup üretilemedi.\n\n**Sebep:** {reason}\n\n"
                f"💡 Lütfen API kotanızı kontrol edin veya birkaç dakika sonra tekrar deneyin."
            )

    st.markdown("---")
    render_why_panel()


main()