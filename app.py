tại sao mình dù có để nguyên liệu lên 10kg thì công thức cũng k thay đỏi?

Đúng, mình đã tìm ra nguyên nhân: mã hiện tại đang giới hạn Azolla tối đa chỉ 1,10 kg, dù bạn nhập 10 kg:

min(1.10, azolla_co)

Các nguyên liệu khác cũng bị giới hạn:

Vỏ chuối: tối đa 0,06 kg
Đậu nành: tối đa 0,18 kg
Bã đậu hũ: tối đa 0,18 kg
Rỉ mật: tối đa 0,06 kg
Nước: tối đa 2,5 lít

Ngoài ra, chương trình luôn dùng cùng một số ngẫu nhiên 123, nên khi lượng nhập vượt các giới hạn trên, nó tạo lại đúng cùng 1.000 công thức và cho ra kết quả giống hệt nhau.

Nói đơn giản: ô nhập hiện tại chỉ mang ý nghĩa “tôi có đủ nguyên liệu hay không”, chưa có chức năng tự mở rộng công thức theo toàn bộ lượng nguyên liệu bạn có. Đây là hạn chế của phiên bản hiện tại, không phải lỗi của Render.

Cách sửa hợp lý nhất là:

AI tiếp tục chọn tỷ lệ phối trộn tối ưu trong phạm vi dữ liệu đã huấn luyện.
Sau đó ứng dụng tự nhân toàn bộ công thức lên theo lượng Azolla người dùng nhập.
Nếu nguyên liệu khác không đủ, ứng dụng báo rõ nguyên liệu đang thiếu.

Ví dụ phương án hiện tại dùng 0,912 kg Azolla. Nếu bạn có và muốn dùng đủ 10 kg Azolla, hệ số nhân là:

$$ 10 \div 0,912 \approx 10,96 $$

Công thức sau khi mở rộng sẽ xấp xỉ:

Azolla: 10 kg
Vỏ chuối: 0,406 kg
Đậu nành: 1,041 kg
Bã đậu hũ: 0,395 kg
Rỉ mật: 0,581 kg
Nước: 24,75 lít
ABABIO: 985 g

Tuy nhiên, app phải kiểm tra bạn có đủ từng nguyên liệu trước khi đề xuất. Mình khuyên sửa thêm lựa chọn:

Dùng tối đa nguyên liệu hiện có
Chọn lượng Azolla muốn ủ

Như vậy công thức sẽ thực sự thay đổi theo quy mô 1 kg, 10 kg hoặc 100 kg, thay vì luôn dừng ở khoảng 1 kg như hiện nay.

được

Mình đã sửa đúng trên bản app.py đang chạy giao diện điện thoại:

Tải app.py đã sửa

Bản mới sẽ:

Nhập 10 kg Azolla → mỗi phương án sử dụng đúng 10 kg Azolla.
Tự nhân các nguyên liệu khác theo đúng tỷ lệ AI đề xuất.
Báo “Đủ nguyên liệu” hoặc liệt kê lượng cần bổ sung.
Giữ nguyên giao diện, hình ảnh và đường link Render hiện tại.
Chỉ mở rộng tỷ lệ; không bắt mô hình dự đoán ngoài phạm vi dữ liệu đã học.

Cách cập nhật:

Tải file trên về máy.
Mở repository azoloop-app trên GitHub.
Chọn Add file → Upload files.
Tải file app.py mới lên để thay thế file cũ.
Nhấn Commit changes.
Chờ Render tự triển khai lại khoảng vài phút.

Bạn không cần thay file mô hình, ảnh hay requirements.txt. Sau khi Render báo Live, mở lại AzoLoop và thử nhập 10 kg Azolla.

app.py
Code

Bạn đang ở mục tạo file mới, nên chưa đúng chỗ nhé.

Làm như sau:

Nhấn Cancel changes ở góc trên bên phải.
Trong danh sách bên trái, nhấn vào file app.py.
Nhấn biểu tượng cây bút ✏️ Edit this file.
Nhấn Ctrl + A trong vùng mã → xóa nội dung cũ.
Mở file app.py mình gửi, nhấn Ctrl + A → Ctrl + C.
Quay lại GitHub và nhấn Ctrl + V.
Nhấn Commit changes… → tiếp tục nhấn Commit changes.

Cách dễ hơn: sau khi hủy trang này, chọn dấu + cạnh nhánh main → Upload files, rồi kéo file app.py mới vào. GitHub sẽ thay thế file cũ khi bạn commit. Không nhập app.py vào ô Name your file, vì cách đó đang tạo thêm file mới.

app.py
999
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89


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


    if loai_vi_sinh == "ABABIO":
        lieu_thap = 70
        lieu_cao = 100
    else:
        lieu_thap = 10
        lieu_cao = 20


    rng = np.random.default_rng(123)
    so_cong_thuc = 1000


    ung_vien = pd.DataFrame({
        "azolla_kg": rng.uniform(
            0.70,
            1.10,
            so_cong_thuc
        ),


        "chuoi_kg": rng.uniform(
            0.02,
            0.06,
            so_cong_thuc
        ),


        "dau_nanh_kg": rng.uniform(
            0,
            0.18,
