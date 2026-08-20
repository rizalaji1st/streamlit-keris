# 🗡️ Klasifikasi & Segmentasi Keris

Aplikasi [Streamlit](https://streamlit.io) untuk mengunggah foto keris, lalu mengenali
**dhapur** (klasifikasi) sekaligus memetakan bentuk bilahnya (**segmentasi**)
menggunakan model YOLO11s-seg dari [Ultralytics](https://docs.ultralytics.com).

## Fitur

- Unggah satu atau beberapa foto sekaligus (JPG, PNG, BMP, WEBP) — tiap foto punya tab sendiri.
- **Segmentasi**: overlay mask semi-transparan berdampingan dengan foto asli, warna berbeda per kelas.
- **Klasifikasi**: dhapur dengan confidence tertinggi, plus rincian confidence semua kelas yang terdeteksi.
- **Objek terpotong**: keris dipisahkan dari latar sebagai PNG transparan.
- **Data**: tabel per objek (kelas, confidence, luas mask, bounding box) dan ekspor JSON.
- Parameter inferensi bisa diatur langsung: confidence threshold, IoU (NMS), ukuran inferensi,
  ketebalan warna mask, tampil/sembunyikan box & label.

## Model

`best.pt` — YOLO11s-seg hasil fine-tuning (Ultralytics 8.3.33, 100 epoch, `imgsz=640`)
pada dataset `dhapur_keris_640-6`, dengan 5 kelas dhapur:

| ID | Kelas |
|----|-------------|
| 0  | brojol |
| 1  | jalak |
| 2  | naga sasra |
| 3  | sengkelat |
| 4  | tilam upih |

Nama kelas dibaca langsung dari file bobot, jadi mengganti `best.pt` dengan hasil
training lain tidak memerlukan perubahan kode.

## Menjalankan secara lokal

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`. Pada instalasi pertama, `ultralytics`
akan menarik PyTorch (unduhan cukup besar). Inferensi berjalan di CPU secara default;
jika tersedia GPU CUDA, Ultralytics memakainya otomatis.

Pengguna macOS/Linux mengaktifkan virtualenv dengan `source .venv/bin/activate`.

## Deploy ke Streamlit Community Cloud

1. Push repositori ini ke GitHub (`best.pt` ikut ter-commit, ±20 MB).
2. Buat app baru di [share.streamlit.io](https://share.streamlit.io), arahkan ke `app.py`.
3. `requirements.txt` dan `packages.txt` sudah disiapkan — `packages.txt` memasang
   dependensi sistem yang dibutuhkan OpenCV.

## Struktur proyek

```
streamlit-keris/
├── app.py             # seluruh aplikasi Streamlit
├── best.pt            # bobot YOLO11s-seg terlatih
├── requirements.txt   # dependensi Python
├── packages.txt       # dependensi sistem untuk Streamlit Cloud
└── README.md
```

## Catatan

- Foto berukuran lebih dari 1600 px otomatis dikecilkan sebelum inferensi agar tetap ringan.
- Model dimuat sekali lalu di-cache (`st.cache_resource`), sehingga hanya request pertama yang lambat.
- Hasil prediksi bergantung pada kualitas foto: bilah terlihat penuh, pencahayaan merata,
  dan latar tidak ramai akan memberi hasil paling baik.
