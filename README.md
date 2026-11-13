<h1> Hotel Booking Cancellation Prediction with Machine Learning </h1>

## 1. Project Overview

Proyek ini menganalisis data reservasi sebuah hotel di Portugal untuk menggali insight, meningkatkan pengambilan keputusan, dan mengidentifikasi tren utama. Fokus utamanya adalah memprediksi probabilitas setiap reservasi dibatalkan (is_canceled) agar hotel dapat mengantisipasi kamar kosong, mengambil tindakan pencegahan pada tamu berisiko tinggi, dan memahami faktor apa saja yang mendorong pembatalan. Sekitar 37% reservasi pada data ini berakhir dibatalkan, sehingga menyebabkan hilangnya pendapatan serta perencanaan okupansi, staf, dan stok yang tidak akurat. Tanpa dasar yang jelas, overbooking juga berisiko membuat tamu yang benar-benar datang harus direlokasi.

Key Objectives:

Objective 1: Menganalisis karakteristik reservasi (jenis deposit, segmen pasar, asal tamu, permintaan khusus, dll.) yang berkaitan dengan pembatalan
Objective 2: Membangun model klasifikasi biner (Random Forest) untuk memprediksi probabilitas pembatalan, dengan target F2-score ≥ 0,75 dan recall kelas batal ≥ 80% pada data test
Objective 3: Menurunkan total biaya kesalahan prediksi (False Positive = biaya relokasi tamu, False Negative = pendapatan kamar yang hilang) dibanding skenario tanpa model, serta memberikan rekomendasi bisnis berdasarkan probabilitas pembatalan

## 2. Data Sources
data_hotel_booking_demand.csv – Dataset utama: 83.573 reservasi sebuah hotel di Portugal dengan 10 fitur dan target is_canceled
Hotel booking demand (hotel_bookings.csv) – Sumber yang sama dengan dataset utama dengan kolom tambahan. Dipakai untuk menghitung harga kamar rata-rata (ADR) dan lama menginap sebagai dasar biaya kesalahan prediksi, serta mengukur seberapa besar fitur yang tidak tersedia (misalnya lead time) membatasi performa model
Hotel Reservations Dataset (Hotel Reservations.csv) – 36.275 reservasi dari hotel lain. Dipakai untuk memvalidasi apakah pola pembatalan yang ditemukan di EDA juga berlaku di hotel lain
Country Mapping – ISO, Continent, Region (continents2.csv, Kaggle) – Pemetaan kode negara ISO ke benua dan sub-region. Dipakai untuk feature engineering, yaitu meringkas country (162 negara) menjadi sub_region

## 3. Technologies Used
Programming Language: Python (Pandas, NumPy)
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn (Pipeline, ColumnTransformer, RandomizedSearchCV, Random Forest, dan model klasifikasi lainnya), SHAP
Others: Jupyter Notebook / Google Colab, Pickle

## 4. Project Structure
├── Hotel_Booking_Cancellation_Prediction_v2.ipynb  <- Notebook analisis dan pemodelan
├── data_hotel_booking_demand.csv                   <- Dataset utama
├── hotel_bookings.csv                              <- Dataset pendukung 1
├── Hotel Reservations.csv                          <- Dataset pendukung 2
├── continents2.csv                                 <- Dataset pendukung 3
└── model_hotel_cancellation.pkl                    <- Model final, threshold, dan daftar fitur

## 5. Summary of Finding

### 5.1 Business Insight
Model final: Random Forest (hasil tuning, class_weight="balanced", threshold 0,37) dengan 10 fitur reservasi. Pada data test (16.715 booking) model mencapai F2-score 0,814, recall 88,9%, precision 60,7%, ROC AUC 0,888, dan PR AUC 0,843. Semua target Project Charter tercapai.
Dampak bisnis: Total biaya turun dari ≈ EUR 2,20 juta (tanpa model) menjadi ≈ EUR 0,61 juta (dengan model), atau hemat ≈ EUR 1,59 juta (±72%), rata-rata ±EUR 95 per booking. Angka ini adalah batas atas karena mengasumsikan semua kamar dari pembatalan yang terdeteksi berhasil dijual ulang. Model juga ±44% lebih murah dibanding strategi ekstrem "anggap semua reservasi batal". Asumsi biaya berasal dari dataset pendukung (FP ≈ EUR 104, FN ≈ EUR 358).
Jenis deposit: Reservasi Non Refund dibatalkan ±99%, jauh di atas No Deposit (±28%). Uang deposit tetap masuk ke hotel, tetapi kamarnya kosong.
Segmen pasar: Groups (±61%) dan Online TA (±37%) paling sering batal. Direct dan Corporate (±15–18%) jauh lebih stabil.
Komitmen tamu: Reservasi tanpa permintaan khusus batal ±48%, turun menjadi ±22% jika ada 1 permintaan. Reservasi yang meminta parkir tidak pernah dibatalkan pada data ini. Pola ini juga terlihat pada data hotel lain.
Asal tamu: Tamu domestik (Portugal) batal ±56%, dibanding ±16–20% untuk GBR, FRA, dan DEU. Pada level sub-region, Southern Europe batal ±50%, sedangkan Western dan Northern Europe ±19–21%.
Riwayat pembatalan: Tamu yang pernah membatalkan cenderung batal lagi (±92–94%). Pola ini khusus untuk hotel ini dan tidak terlihat pada data hotel lain.
Fitur paling penting: market_segment dan deposit_type (ROC AUC turun ±0,10–0,12 jika diacak), diikuti total_of_special_requests, is_domestic, customer_type, dan required_car_parking_spaces.
Keandalan probabilitas: Reservasi dengan probabilitas > 0,8 benar-benar batal ±96%, sedangkan probabilitas ≤ 0,2 hanya batal ±3%. Zona 0,4–0,6 adalah zona ketidakpastian (pembatalan aktual ±36%).
Kondisi model kurang andal: Segmen Direct, Corporate, dan Complementary (recall rendah), tamu Western Europe (recall 55%), serta Online TA (precision ±49%, banyak alarm palsu).
Limitasi data: Dataset utama tidak memiliki lead time, tanggal kedatangan, harga, dan lama menginap. Eksperimen pada hotel_bookings.csv menunjukkan penambahan fitur tersebut menaikkan ROC AUC dari ±0,89 menjadi ±0,94, sehingga batas performa model berasal dari data, bukan dari algoritma.

### 5.2 Actionable Recommendation
Gunakan tindakan bertahap berdasarkan probabilitas pembatalan:
> 0,8: kamar boleh dimasukkan ke kuota overbooking / dijual ulang
0,37 – 0,8: lakukan konfirmasi ulang ke tamu (email/WhatsApp H-7) dan tawarkan fasilitas tambahan. Jangan langsung di-overbook
< 0,37: diperlakukan normal
Lakukan overbooking per tanggal memakai jumlah probabilitas semua reservasi pada tanggal tersebut sebagai perkiraan jumlah kamar yang akan batal, dengan angka overbooking sedikit di bawahnya sebagai margin aman.
Tinjau ulang kerja sama dengan agen/grup yang memakai deposit Non Refund. Pertimbangkan release date (kamar dilepas otomatis jika tidak dikonfirmasi) atau batasi kuota blok kamar.
Kurangi risiko dari segmen Online TA serta tamu domestik/Southern Europe dengan deposit sebagian, pengingat konfirmasi, atau insentif untuk booking langsung (Direct hanya batal ±14%).
Dorong keterlibatan tamu sebelum kedatangan (pilihan kamar, parkir, atau permintaan khusus lewat email pra-kedatangan). Ini adalah hubungan korelasi, sehingga efeknya perlu diuji dengan A/B test.
Perhatikan reservasi yang dibuat jauh hari. Data pendukung menunjukkan reservasi dengan lead time > 6 bulan batal ±55–95%. Selama fitur ini belum masuk model, tambahkan aturan konfirmasi ulang untuk reservasi jangka panjang.
Lakukan uji coba bertahap (A/B test) sebelum dipakai penuh, misalnya model dipakai untuk 50% tanggal lalu bandingkan okupansi dan biaya relokasi tamu dengan 50% tanggal lainnya.
Pengembangan model: tambahkan lead_time, bulan kedatangan, adr, lama menginap, dan jumlah tamu dari PMS hotel (dampak terbesar); konfirmasi waktu pencatatan required_car_parking_spaces untuk memastikan tidak ada data leakage; ganti asumsi biaya (cost_fp, cost_fn) dengan data keuangan hotel yang sebenarnya lalu hitung ulang threshold; coba algoritma boosting lain (XGBoost, LightGBM, CatBoost) dengan kalibrasi probabilitas; pantau recall dan precision setiap bulan serta latih ulang model setiap kuartal.

## 6. Contact
Name: 
- Raffy Reyhan Ramadhana
- Muhammad Aldi Riansyah
- Nazma Aulia

Email:
- raffyreyhan2902@gmail.com
- m.aldiriansyahhh2@gmail.com
- aulianazma67@gmail.com

Linkedin:
- https://www.linkedin.com/in/raffy-reyhan-ramadhana-724070364
- https://www.linkedin.com/in/aldiriansyahhh/
- https://www.linkedin.com/in/nazma-aulia-35520b342/