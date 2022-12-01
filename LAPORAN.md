# Laporan Proyek Machine Learning - Yunus Ahmad Hasan Baktir

## Domain Proyek

Perubahan iklim memaksa kota besar untuk menyesuaikan ulang infrastruktur transportasinya. Konsep berbagi kendaraan seperti *car sharing, bike sharing* menjadi semakin populer. Jika di terapkan dengan baik, cara ini bisa dapat berkontribusi dalam mitigasi perubahan iklim. Khususnya *bike sharing* karena tidak memerlukan bahan bakar untuk moda transportasi ini.

Namun ada masalah yang melekat pada konsep *bike sharing* ini, yaitu permintaan yang bervariasi di setiap stasiun nya yang perlu diseimbangkan untuk menghindari kelebihan atau kekurangan pasokan sepeda.

Dengan memprediksi permintaan di masa yang akan datang diharapkan dapat membantu mengatasi masalah tersebut.

## Referensi
- Using data mining techniques for bike sharing demand prediction in metropolitan city
VE Sathishkumar, J Park, Y Cho - Computer Communications, 2020 - Elsevier

## Business Understanding

Kondisi iklim yang berubah-ubah menjadikan permintaan sepeda yang berbeda-beda di setiap harinya. Permintaan sepeda naik atau turun seiring dengan perubahan iklim.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap permintaan sepeda?
- Berapa jumlah permintaan sepeda dengan karakteristik atau fitur tertentu?  

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui fitur yang paling berkorelasi dengan jumlah permintaan sepeda.
- Membuat model machine learning yang dapat memprediksi jumlah permintaan sepeda seakurat mungkin berdasarkan fitur-fitur yang ada.


## Solution Statements
- Membangun model dengan menggunakan algoritma LSTM 

## Data Understanding
Data yang digunakan adalah dataset *Bike Sharing Washington DC*. Dataset ini dapat digunakan untuk memprediksi permintaan untuk menghindari kelebihan atau kekurangan pasokan. Rentang waktu data dari 1 Januari 2011 hingga 31 Desember 2018.

Dataset: *[Bike Sharing Washington DC](httpshttps://www.kaggle.com/datasets/juliajemals/bike-sharing-washington-dc)*.

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Bike Sharing Washington DC dataset adalah sebagai berikut:
- date : tanggal dengan format yyyy-mm-dd
- temp_avg : suhu harian rata-rata dalam derajat Celcius
- temp_min : suhu harian minimum dalam derajat Celcius
- temp_max : suhu harian maksimum dalam derajat Celcius
- temp_observ : suhu pada saat pengamatan dalam derajat Celcius
- precip : jumlah curah hujan dalam mm
- wind : kecepatan angin dalam meter per detik
- wt_fog : kabut jenis cuaca, kabut es, atau kabut beku (mungkin termasuk kabut tebal)
- wtheavyfog : jenis cuaca kabut tebal atau kabut beku yang naik-turun (tidak selalu dibedakan dari kabut)
- wt_thunder : jenis cuaca guntur
- wt_sleet : butiran es jenis cuaca, hujan es, butiran salju, atau hujan es kecil
- wt_hail : hujan es jenis cuaca (mungkin termasuk hujan es kecil)
- wt_glaze : lapisan atau rima jenis cuaca
- wt_haze : tipe cuaca asap atau kabut
- wtdriftsnow : jenis cuaca yang bertiup atau salju yang melayang
- wthighwind : jenis cuaca angin kencang atau merusak
- wt_mist : kabut jenis cuaca
- wt_drizzle : gerimis jenis cuaca
- wt_rain : tipe cuaca hujan (mungkin termasuk hujan beku, gerimis, dan gerimis beku)
- wtfreezerain : tipe cuaca hujan beku
- wt_snow : salju jenis cuaca, butiran salju, atau kristal es
- wtgroundfog : kabut tanah jenis cuaca
- wticefog : kabut es jenis cuaca atau kabut beku
- wtfreezedrizzle : gerimis beku jenis cuaca
- wt_unknown : tipe cuaca sumber presipitasi tidak diketahui
- casual : jumlah pelanggan yang tidak terdaftar
- registered : jumlah pelanggan terdaftar
- total_cust : jumlah pelanggan terdaftar dan biasa
- holiday : menunjukkan apakah hari itu hari libur atau tidak

Melakukan Visualisasi data untuk melihat korelasi dengan target data
![Alt text](https://raw.githubusercontent.com/yunusffz/ml-terapan-proyek-1/main/images/visualisasi-data.PNG "Title")

![Alt text](https://raw.githubusercontent.com/yunusffz/ml-terapan-proyek-1/main/images/total-cust.PNG "Title")


## Data Preparation
Tahapan data preparation yang dilakukan:
- Penghapusan Kolom yang tidak pakai: untuk mempercepat proses pembelajaran model
- Pengisian Kolom N/A dengan nilainya: Tahapan ini diisi dengan 0 karena bertipe boolean.
- Normalisasi Data: Untuk mereduksi penyimpangan nilai pada rentang yang sama
- Penghapusan Outlier: agar data linear dan meningkatkan akurasi saat pembelajar model

## Modeling
Pada Tahapan ini digunakan model neural network. Karena data yang digunakan merupakan data time-series maka digunakan  LSTM (Long Short Term Memory) untuk pemodel data time series. Digunakan LSTM karena mampu mengatasi ketergantungan jangka panjang (long term dependencies) pada masukannya.

Terdapat tahapan penyesuaian parameter sebagai berikut:
- Melakukan drop sebesar 20% setelah lapisan pertama untuk mengurangi overfitting
- Mengubah hyper parameter menjadi 0.01 

## Evaluation
Metrik yang digunakan pada variabel kontinyu ini digunakan metrik MAE (Margin Absolute Error).
MAE mengukur besarnya rata-rata kesalahan dalam serangkaian prediksi, tanpa mempertimbangkan arahnya. Ini adalah rata-rata sampel uji dari perbedaan absolut antara prediksi dan pengamatan aktual di mana semua perbedaan individu memiliki bobot yang sama.

Pada proyek ini diset MAE sebesar 30%, ketika MAE pada prediksi berada dibawah 30% maka model dianggap sudah sesuai dengan target. Ketika nilai MAE diatas 30% maka akan terus dilakukan pembelajaran sebanyak epoch yang telah di set.
