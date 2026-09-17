
import joblib
import pandas as pd
import numpy as np
import gradio as gr
from pathlib import Path
import base64
import mimetypes
from html import escape
import os


# ==================================================
# 1. ĐỌC RANDOM FOREST
# ==================================================

THU_MUC_APP = Path(__file__).resolve().parent
goi_mo_hinh = joblib.load(THU_MUC_APP / "random_forest_azolla_demo.pkl")

cac_mo_hinh_app = goi_mo_hinh["models"]

def anh_nhung(ten_file):
    duong_dan = THU_MUC_APP / ten_file
    mime = mimetypes.guess_type(duong_dan.name)[0] or "image/png"
    ma_hoa = base64.b64encode(duong_dan.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{ma_hoa}"

LOGO = anh_nhung("logo.png")
AZOLLA = anh_nhung("azolla.jpg")
CHUOI = anh_nhung("chuoi.jpg")
DAU_NANH = anh_nhung("dau_nanh.jpg")
BA_DAU_HU = anh_nhung("ba_dau_hu.jpg")
RI_MAT = anh_nhung("ri_mat.jpg")
NUOC = anh_nhung("nuoc.jpg")


# ==================================================
# 2. HÀM ĐỀ XUẤT CÔNG THỨC
# ==================================================

def de_xuat_cong_thuc(
    azolla_co,
    chuoi_co,
    dau_nanh_co,
    ba_dau_hu_co,
    ri_mat_co,
    nuoc_co,
    loai_vi_sinh,
    vi_sinh_co,
    thoi_gian_u,
    ph_muc_tieu,
    ec_muc_tieu,
    n_muc_tieu,
    p_muc_tieu,
    k_muc_tieu
):

    if azolla_co < 0.70:
        return (
            pd.DataFrame(),
            "Cần tối thiểu 0,70 kg Azolla."
        )

    if chuoi_co < 0.02:
        return (
            pd.DataFrame(),
            "Cần tối thiểu 0,02 kg vỏ chuối."
        )

    if ri_mat_co < 0.035:
        return (
            pd.DataFrame(),
            "Cần tối thiểu 0,035 kg rỉ mật."
        )

    if nuoc_co < 1.50:
        return (
            pd.DataFrame(),
            "Cần tối thiểu 1,50 lít nước."
        )

    if loai_vi_sinh == "ABABIO":
        lieu_thap = 70
        lieu_cao = min(100, vi_sinh_co)
    else:
        lieu_thap = 10
        lieu_cao = min(20, vi_sinh_co)

    if lieu_cao < lieu_thap:
        return (
            pd.DataFrame(),
            f"Không đủ {loai_vi_sinh}."
        )

    rng = np.random.default_rng(123)
    so_cong_thuc = 1000

    ung_vien = pd.DataFrame({
        "azolla_kg": rng.uniform(
            0.70,
            min(1.10, azolla_co),
            so_cong_thuc
        ),

        "chuoi_kg": rng.uniform(
            0.02,
            min(0.06, chuoi_co),
            so_cong_thuc
        ),

        "dau_nanh_kg": rng.uniform(
            0,
            min(0.18, dau_nanh_co),
            so_cong_thuc
        ),

        "ba_dau_hu_kg": rng.uniform(
            0,
            min(0.18, ba_dau_hu_co),
            so_cong_thuc
        ),

        "ri_mat_kg": rng.uniform(
            0.035,
            min(0.060, ri_mat_co),
            so_cong_thuc
        ),

        "nuoc_lit": rng.uniform(
            1.50,
            min(2.50, nuoc_co),
            so_cong_thuc
        ),

        "loai_vi_sinh": [
            loai_vi_sinh
        ] * so_cong_thuc,

        "vi_sinh_g": rng.uniform(
            lieu_thap,
            lieu_cao,
            so_cong_thuc
        ),

        "thoi_gian_u_ngay": [
            int(thoi_gian_u)
        ] * so_cong_thuc
    })

    # Random Forest dự đoán
    for dau_ra, mo_hinh in cac_mo_hinh_app.items():
        ung_vien[dau_ra] = mo_hinh.predict(
            ung_vien
        )

    # Tính điểm sai lệch
    ung_vien["diem"] = (
        abs(ung_vien["ph_sau_u"] - ph_muc_tieu)
        / max(ph_muc_tieu, 0.01)

        + abs(
            ung_vien["ec_sau_u_ms_cm"]
            - ec_muc_tieu
        ) / max(ec_muc_tieu, 0.01)

        + abs(
            ung_vien["n_tong_mg_l"]
            - n_muc_tieu
        ) / max(n_muc_tieu, 0.01)

        + abs(
            ung_vien["p_mg_l"]
            - p_muc_tieu
        ) / max(p_muc_tieu, 0.01)

        + abs(
            ung_vien["k_mg_l"]
            - k_muc_tieu
        ) / max(k_muc_tieu, 0.01)
    )

    # Đổi tên cột cho dễ đọc
    doi_ten_cot = {
        "azolla_kg": "Azolla (kg)",
        "chuoi_kg": "Vỏ chuối (kg)",
        "dau_nanh_kg": "Đậu nành (kg)",
        "ba_dau_hu_kg": "Bã đậu hũ (kg)",
        "ri_mat_kg": "Rỉ mật (kg)",
        "nuoc_lit": "Nước (lít)",
        "loai_vi_sinh": "Vi sinh",
        "vi_sinh_g": "Liều vi sinh (g)",
        "thoi_gian_u_ngay": "Thời gian ủ (ngày)",
        "ph_sau_u": "pH dự đoán",
        "ec_sau_u_ms_cm": "EC dự đoán",
        "n_tong_mg_l": "N dự đoán (mg/L)",
        "p_mg_l": "P dự đoán (mg/L)",
        "k_mg_l": "K dự đoán (mg/L)",
        "diem": "Điểm sai lệch"
    }

    ket_qua = (
        ung_vien
        .sort_values("diem")
        .head(3)
        .rename(columns=doi_ten_cot)
        .round(3)
        .reset_index(drop=True)
    )

    ket_qua.insert(
        0,
        "Phương án",
        ["Phương án 1", "Phương án 2", "Phương án 3"]
    )

    ghi_chu = (
        "**Lưu ý:** Kết quả được tạo từ dữ liệu mô phỏng. "
        "Chưa được xác nhận bằng kết quả phân tích lab."
    )

    return ket_qua, ghi_chu


CSS = """
html, body { background:#eef3ef !important; color-scheme:light !important; }
.gradio-container {
  max-width:460px !important; margin:auto !important; padding:0 !important;
  color-scheme:light !important;
  --body-background-fill:#ffffff !important;
  --block-background-fill:#ffffff !important;
  --input-background-fill:#ffffff !important;
  --block-border-color:#dfe8e1 !important;
  --body-text-color:#173126 !important;
  --block-label-text-color:#173126 !important;
}
.phone { min-height:100vh; background:#fff !important; padding:18px; color:#173126 !important; }
.topbar { display:flex; align-items:center; gap:10px; margin-bottom:18px; }
.logo-img { width:58px; height:58px; object-fit:contain; border-radius:50%; }
.brand { font-size:23px; font-weight:850; color:#075b32 !important; -webkit-text-fill-color:#075b32 !important; }
.sub { color:#68756e !important; -webkit-text-fill-color:#68756e !important; font-size:13px; }
.hero { height:280px; border-radius:20px; position:relative; overflow:hidden; margin:18px 0; background:#d9edd9; }
.hero img { width:100%; height:100%; object-fit:cover; display:block; }
.hero-overlay { position:absolute; inset:auto 0 0 0; padding:36px 18px 16px; background:linear-gradient(transparent,rgba(0,55,25,.82)); color:white !important; }
.hero-overlay b,.hero-overlay div { color:white !important; -webkit-text-fill-color:white !important; }
.title { color:#084f2d !important; -webkit-text-fill-color:#084f2d !important; font-size:27px; font-weight:850; line-height:1.15; margin:8px 0; }
.desc { color:#506158 !important; -webkit-text-fill-color:#506158 !important; line-height:1.5; }
.badge { background:#edf7ef; border-radius:14px; padding:13px; color:#075b32 !important; -webkit-text-fill-color:#075b32 !important; font-weight:700; margin:14px 0; }
.step { color:#15803d !important; -webkit-text-fill-color:#15803d !important; font-weight:800; float:right; }
.screen-title { color:#103d28 !important; -webkit-text-fill-color:#103d28 !important; font-size:25px; font-weight:850; margin:8px 0 4px; }
.screen-desc { color:#68756e !important; -webkit-text-fill-color:#68756e !important; margin-bottom:16px; }
.ingredient-card { border:1px solid #dfe8e1 !important; border-radius:17px !important; padding:10px !important; background:#fff !important; gap:7px !important; }
.ingredient-photo { height:122px; overflow:hidden; background:#f1f7f2; border-radius:12px; }
.ingredient-photo img { width:100%; height:100%; object-fit:cover; display:block; }
.ingredient-name { color:#173126 !important; -webkit-text-fill-color:#173126 !important; font-size:15px; font-weight:800; text-align:center; margin-top:7px; }
.ingredient-card .form, .ingredient-card .block { border:none !important; box-shadow:none !important; background:#fff !important; padding:0 !important; }
.ingredient-card label { font-weight:700 !important; }
.card { border:1px solid #e0e8e2; border-radius:16px; padding:14px; margin-bottom:12px; background:#fff; }
.tip { background:#edf7ef; border-radius:15px; padding:14px; color:#225c38; margin:15px 0; }
.prep-note { background:#fff7e8; border:1px solid #f4dfb7; border-radius:15px; padding:14px; color:#654b16 !important; margin:15px 0; line-height:1.45; }
.warning { background:#fff4df; color:#7a5400; border-radius:13px; padding:12px; margin-top:12px; }
.gradio-container button.primary { background:#08783f !important; border:none !important; min-height:54px; border-radius:14px !important; font-weight:800; }
.gradio-container button.secondary { min-height:50px; border-radius:14px !important; }
.gradio-container, .gradio-container label, .gradio-container span, .gradio-container p { color:#173126 !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
  color:#17251d !important; -webkit-text-fill-color:#17251d !important;
  background:#fff !important; border-color:#cfdad2 !important;
}
.gradio-container .block, .gradio-container .form { background:#fff !important; color:#173126 !important; }
.result-wrap { display:flex; flex-direction:column; gap:14px; }
.result-card { border:1px solid #dce8df; border-radius:18px; padding:16px; background:#fff; box-shadow:0 5px 18px rgba(9,91,50,.07); }
.result-card.best { border:2px solid #0a8749; background:#f7fcf8; }
.result-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; color:#075b32 !important; font-weight:850; font-size:18px; }
.best-tag { font-size:11px; background:#0a8749; color:#fff !important; -webkit-text-fill-color:#fff !important; border-radius:999px; padding:5px 8px; }
.result-section { color:#6a776f !important; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.04em; margin:10px 0 6px; }
.result-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px 12px; }
.result-item { display:flex; justify-content:space-between; gap:8px; border-bottom:1px solid #edf2ee; padding:6px 0; font-size:13px; color:#243c30 !important; }
.result-item b { color:#075b32 !important; white-space:nowrap; }
footer { display:none !important; }
"""

HEADER = f"""
<div class='topbar notranslate' translate='no'>
  <img class='logo-img' src='{LOGO}' alt='Logo Azo-MAPP'>
  <div><div class='brand'>AzoLoop</div><div class='sub'>Azo-MAPP Team</div></div>
</div>
"""

def doi_man_hinh(so):
    return [gr.update(visible=(i == so)) for i in range(5)]

def tao_ket_qua(*args):
    du_lieu = list(args)
    che_do_chi_tieu = du_lieu.pop(9)
    dung_tham_chieu = che_do_chi_tieu == "Dùng bộ tham chiếu mô phỏng"
    if dung_tham_chieu:
        # Bộ chỉ tiêu chỉ dùng để minh họa khi người dùng chưa có kết quả lab.
        du_lieu[9:14] = [6.3, 2.0, 620, 135, 240]
    bang, ghi_chu = de_xuat_cong_thuc(*du_lieu)
    if dung_tham_chieu:
        ghi_chu += "\n\n**Chế độ đang dùng:** Bộ chỉ tiêu tham chiếu mô phỏng (pH 6,3; EC 2,0 mS/cm; N 620; P 135; K 240 mg/L)."
    if bang.empty:
        html = "<div class='warning'>Không thể tạo công thức. Vui lòng kiểm tra lại lượng nguyên liệu.</div>"
    else:
        nguyen_lieu = ["Azolla (kg)", "Vỏ chuối (kg)", "Đậu nành (kg)", "Bã đậu hũ (kg)", "Rỉ mật (kg)", "Nước (lít)", "Vi sinh", "Liều vi sinh (g)", "Thời gian ủ (ngày)"]
        du_doan = ["pH dự đoán", "EC dự đoán", "N dự đoán (mg/L)", "P dự đoán (mg/L)", "K dự đoán (mg/L)"]
        cac_the = []
        for i, dong in bang.iterrows():
            lop = "result-card best" if i == 0 else "result-card"
            nhan = "<span class='best-tag'>Gần mục tiêu nhất</span>" if i == 0 else ""
            muc_nguyen_lieu = "".join(
                f"<div class='result-item'><span>{escape(cot)}</span><b>{escape(str(dong[cot]))}</b></div>"
                for cot in nguyen_lieu
            )
            muc_du_doan = "".join(
                f"<div class='result-item'><span>{escape(cot)}</span><b>{escape(str(dong[cot]))}</b></div>"
                for cot in du_doan
            )
            cac_the.append(f"""
              <div class='{lop}'>
                <div class='result-head'><span>{escape(str(dong['Phương án']))}</span>{nhan}</div>
                <div class='result-section'>Thành phần đề xuất</div><div class='result-grid'>{muc_nguyen_lieu}</div>
                <div class='result-section'>Chỉ tiêu dự đoán</div><div class='result-grid'>{muc_du_doan}</div>
              </div>""")
        html = "<div class='result-wrap'>" + "".join(cac_the) + "</div>"
    return (*doi_man_hinh(4), html, ghi_chu)

JS = """
function () {
  document.documentElement.lang = 'vi';
  document.documentElement.setAttribute('translate', 'no');
  document.body.setAttribute('translate', 'no');
  document.body.classList.add('notranslate');
}
"""

with gr.Blocks(
    title="AzoLoop – Trợ lý phối trộn phân bón Azolla",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="green"),
    head='<meta name="google" content="notranslate"><meta http-equiv="Content-Language" content="vi"><meta charset="utf-8">',
    js=JS
) as app:
    with gr.Column(visible=True, elem_classes=["phone"]) as man_0:
        gr.HTML(HEADER)
        gr.HTML(f"""
        <div class='title' style='text-align:center'>Đề xuất công thức phân bón Azolla bằng AI</div>
        <div class='hero'><img src='{AZOLLA}' alt='Bèo Azolla'><div class='hero-overlay'><b>Bèo Azolla</b><div>Nguồn dinh dưỡng xanh cho nông nghiệp bền vững</div></div></div>
        """)
        bat_dau = gr.Button("BẮT ĐẦU PHỐI TRỘN  →", variant="primary")
        gr.HTML("<div class='warning'>ⓘ Kết quả AI chỉ mang tính tham khảo và đang sử dụng dữ liệu mô phỏng.</div>")

    with gr.Column(visible=False, elem_classes=["phone"]) as man_1:
        gr.HTML(HEADER + "<span class='step'>1/3</span><div class='screen-title'>1. Nguyên liệu hiện có</div><div class='screen-desc'>Nhập khối lượng tối đa bạn có thể sử dụng.</div>")
        with gr.Row():
            with gr.Column(elem_classes=["ingredient-card"]):
                gr.HTML(f"<div class='ingredient-photo'><img src='{AZOLLA}' alt='Bèo Azolla'></div><div class='ingredient-name'>Bèo Azolla</div>")
                azolla_input = gr.Number(label="Khối lượng (kg)", value=1.10, minimum=0)
            with gr.Column(elem_classes=["ingredient-card"]):
                gr.HTML(f"<div class='ingredient-photo'><img src='{CHUOI}' alt='Vỏ chuối'></div><div class='ingredient-name'>Vỏ chuối</div>")
                chuoi_input = gr.Number(label="Khối lượng (kg)", value=0.06, minimum=0)
        with gr.Row():
            with gr.Column(elem_classes=["ingredient-card"]):
                gr.HTML(f"<div class='ingredient-photo'><img src='{DAU_NANH}' alt='Đậu nành'></div><div class='ingredient-name'>Đậu nành</div>")
                dau_nanh_input = gr.Number(label="Khối lượng (kg)", value=0.18, minimum=0)
            with gr.Column(elem_classes=["ingredient-card"]):
                gr.HTML(f"<div class='ingredient-photo'><img src='{BA_DAU_HU}' alt='Bã đậu hũ'></div><div class='ingredient-name'>Bã đậu hũ</div>")
                ba_dau_input = gr.Number(label="Khối lượng (kg)", value=0.18, minimum=0)
        with gr.Row():
            with gr.Column(elem_classes=["ingredient-card"]):
                gr.HTML(f"<div class='ingredient-photo'><img src='{RI_MAT}' alt='Rỉ mật'></div><div class='ingredient-name'>Rỉ mật</div>")
                ri_mat_input = gr.Number(label="Khối lượng (kg)", value=0.06, minimum=0)
            with gr.Column(elem_classes=["ingredient-card"]):
                gr.HTML(f"<div class='ingredient-photo'><img src='{NUOC}' alt='Nước'></div><div class='ingredient-name'>Nước</div>")
                nuoc_input = gr.Number(label="Thể tích (lít)", value=2.50, minimum=0)
        gr.HTML("<div class='prep-note'>✂️ <b>Khuyến nghị:</b> Nên cắt hoặc xay nhỏ bèo Azolla, chuối, đậu nành và bã đậu hũ trước khi ủ để tăng diện tích tiếp xúc và hỗ trợ quá trình phân giải.</div>")
        with gr.Row():
            ve_home_1 = gr.Button("‹ QUAY LẠI", variant="secondary")
            den_dk = gr.Button("TIẾP TỤC  →", variant="primary")

    with gr.Column(visible=False, elem_classes=["phone"]) as man_2:
        gr.HTML(HEADER + "<span class='step'>2/3</span><div class='screen-title'>2. Điều kiện ủ</div><div class='screen-desc'>Thiết lập các thông số ủ phân bón.</div>")
        loai_vi_sinh_input = gr.Radio(["ABABIO", "EM"], value="ABABIO", label="Loại vi sinh")
        vi_sinh_input = gr.Number(label="Liều vi sinh hiện có (g)", value=100, minimum=0)
        thoi_gian_input = gr.Slider(12, 16, value=14, step=1, label="Thời gian ủ (ngày)")
        gr.HTML("<div class='tip'>💡 Thời gian ủ từ 12–16 ngày được dùng trong dữ liệu mô phỏng hiện tại.</div>")
        with gr.Row():
            ve_nl = gr.Button("‹ QUAY LẠI", variant="secondary")
            den_mt = gr.Button("TIẾP TỤC  →", variant="primary")

    with gr.Column(visible=False, elem_classes=["phone"]) as man_3:
        gr.HTML(HEADER + "<span class='step'>3/3</span><div class='screen-title'>3. Chỉ tiêu mong muốn</div><div class='screen-desc'>Nhập các chỉ tiêu dinh dưỡng cần đạt gần nhất.</div>")
        dung_tham_chieu_input = gr.Radio(
            choices=["Tự nhập chỉ tiêu", "Dùng bộ tham chiếu mô phỏng"],
            value="Tự nhập chỉ tiêu",
            label="Chọn cách xác định chỉ tiêu",
            elem_classes=["notranslate", "reference-mode"]
        )
        gr.HTML("<div class='tip notranslate' translate='no'>Nếu chưa có kết quả phân tích hoặc mục tiêu cụ thể, chọn <b>Dùng bộ tham chiếu mô phỏng</b>. Khi đó các giá trị bên dưới sẽ không được dùng để tính toán.</div>")
        ph_input = gr.Number(label="pH", value=6.3)
        ec_input = gr.Number(label="EC (mS/cm)", value=2.0)
        n_input = gr.Number(label="N (mg/L)", value=620)
        p_input = gr.Number(label="P (mg/L)", value=135)
        k_input = gr.Number(label="K (mg/L)", value=240)
        with gr.Row():
            ve_dk = gr.Button("‹ QUAY LẠI", variant="secondary")
            nut_de_xuat = gr.Button("🧪 ĐỀ XUẤT CÔNG THỨC", variant="primary")

    with gr.Column(visible=False, elem_classes=["phone"]) as man_4:
        gr.HTML(HEADER + "<div class='screen-title'>Công thức đề xuất</div><div class='screen-desc'>Ba công thức có chỉ tiêu dự đoán gần mục tiêu nhất.</div>")
        bang_ket_qua = gr.HTML()
        ghi_chu_output = gr.Markdown(elem_classes=["warning"])
        with gr.Row():
            lam_lai = gr.Button("‹ THAY ĐỔI", variant="secondary")
            ve_home_2 = gr.Button("TRANG CHỦ", variant="primary")

    cac_man = [man_0, man_1, man_2, man_3, man_4]
    bat_dau.click(lambda: doi_man_hinh(1), outputs=cac_man)
    ve_home_1.click(lambda: doi_man_hinh(0), outputs=cac_man)
    den_dk.click(lambda: doi_man_hinh(2), outputs=cac_man)
    ve_nl.click(lambda: doi_man_hinh(1), outputs=cac_man)
    den_mt.click(lambda: doi_man_hinh(3), outputs=cac_man)
    ve_dk.click(lambda: doi_man_hinh(2), outputs=cac_man)
    lam_lai.click(lambda: doi_man_hinh(3), outputs=cac_man)
    ve_home_2.click(lambda: doi_man_hinh(0), outputs=cac_man)
    nut_de_xuat.click(
        fn=tao_ket_qua,
        inputs=[azolla_input, chuoi_input, dau_nanh_input, ba_dau_input,
                ri_mat_input, nuoc_input, loai_vi_sinh_input, vi_sinh_input,
                thoi_gian_input, dung_tham_chieu_input,
                ph_input, ec_input, n_input, p_input, k_input],
        outputs=cac_man + [bang_ket_qua, ghi_chu_output]
    )


# ==================================================
# 4. CHẠY APP
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
