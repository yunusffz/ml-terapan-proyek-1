# Laporan Proyek Machine Learning - Yunus Ahmad Hasan Baktir

## Domain Proyek

Perubahan iklim memaksa kota besar untuk menyesuaikan ulang infrastruktur transportasinya. Konsep berbagi kendaraan seperti car sharing, bike sharing menjadi semakin populer. Jika di terapkan dengan baik, cara ini bisa dapat berkontribusi dalam mitigasi perubahan iklim. Khususnya bike sharing karena tidak memerlukan bahan bakar untuk moda transportasi ini. Apa itu bike sharing ? Bike sharing adalah penyewaan sepeda yang ditempatkan pada lokasi-lokasi tertentu ditempat umum yang dapat digunakan oleh masyarakat. 

Jumlah permintaan sewa sepeda bervariasi, tergantung dari kondisi cuaca yang memungkinkan untuk bersepeda. Stok sepeda menjadi penting untuk memenuhi tingkat permintaan sewa sepeda pada kasus ini. Untuk dapat memenuhi permintaan sesuai dengan jumlah permintaan sepeda agar tidak terjadi kelebihan/kekurangan stok sepeda, maka perlu dibuatkan sebuah model untuk dapat memprediksi jumlah permintaan sepeda di masa yang akan datang.

## Business Understanding

Permintaan sepeda sangat bergantung dengan kondisi iklim yang cocok untuk bersepeda yang terkadang tinggi dan terkadang rendah. Perlu adanya model yang dapat memprediksi jumlah permintaan sepeda agar stok sepeda bisa disesuaikan dengan jumlah permintaan sepeda di setiap harinya. 

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Fitur apa saja yang mempengaruhi permintaan sepeda ?
- Model apa yang cocok untuk dapat memprediksi permintaan sepeda ?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengetahui fitur yang paling berkorelasi dengan jumlah permintaan sepeda.
- Membuat model machine learning yang dapat memprediksi jumlah permintaan sepeda seakurat mungkin berdasarkan fitur-fitur yang ada.


## Solution Statements
- Membangun model dengan menggunakan algoritma LSTM, digunakannya algoritma ini untuk dapat memprediksi salah satu variabel yaitu jumlah pelanggan (penyewa sepeda) pada hari-hari tertentu. Algoritma ini cocok digunakan pada kasus ini karena data yang digunakan merupakan data time series. Dimana prediksi terbaik yang diambil didasarkan pada tingkat kesalahan prediksi, semakin kecil tingkat kesalahan, maka semakin tepat pula model dalam memprediksi.

## Data Understanding
Data yang digunakan adalah dataset *Bike Sharing Washington DC*. 
### Tentang Dataset

#### Konteks
Perubahan iklim memaksa kota untuk mencitrakan ulang infrastruktur transportasi mereka. Konsep mobilitas bersama, seperti berbagi mobil, berbagi sepeda, atau berbagi skuter menjadi semakin populer. Dan jika diterapkan dengan baik, mereka benar-benar dapat berkontribusi dalam mitigasi perubahan iklim. Berbagi sepeda khususnya menarik karena tidak diperlukan listrik bensin (kecuali e-sepeda digunakan) untuk moda transportasi ini. Namun, ada masalah yang melekat pada jenis mobilitas bersama ini:

- Permintaan yang bervariasi di stasiun bike sharing perlu diseimbangkan untuk menghindari kelebihan pasokan atau kekurangan
- Sepeda yang banyak digunakan lebih sering rusak

Prediksi permintaan masa depan dapat membantu mengatasi masalah tersebut. Selain itu, perkiraan permintaan dapat membantu operator memutuskan apakah akan mengembangkan bisnis, menentukan harga yang memadai, dan menghasilkan pendapatan tambahan melalui iklan di stasiun yang sangat sibuk.

Link Dataset: *[Bike Sharing Washington DC](httpshttps://www.kaggle.com/datasets/juliajemals/bike-sharing-washington-dc)*.
### Metadata
Jumlah baris data: 2922 baris
Rentang waktu: 1 Januari 2011 hingga 31 Desember 2018.

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
![Alt text](https://raw.githubusercontent.com/yunusffz/ml-terapan-proyek-1/main/images/temp-observe.PNG "Title")
![Alt text](https://raw.githubusercontent.com/yunusffz/ml-terapan-proyek-1/main/images/total-cust.PNG "Title")

Dari hasil visualisasi berikut didapat informasi bahwa total customer punya pola yang mirip dengan temp observe. Dapat dikatakan temp observe mempunyai keterkaitan dengan total customer.


## Data Preparation
Tahapan data preparation yang dilakukan:
### Data Cleaning
Agar data dapat diolah dengan baik, nilai yang berupa N/A akan diisikan dengan nilai 0. Nilai tersebut disesuaikan dengan fitur atributnya.
### Feature Selection
Pemilihan fitur yang digunakan untuk proses pembelajaran model. Proses ini perlu dilakukan dengan cara menghapus kolom yang tidak digunakan atau tidak terkait dengan target data yang kita cari. Tujuannya untuk mencegah fitur yang tidak relevan menghambat proses pembelajaran model.

### Feature Engineering
Pada tahap ini digunakan MinMaxScaler untuk menyamakan rentang nilai setiap fitur. Cara kerjanya setiap nilai pada sebuah fitur dikurangi dengan nilai minimum fitur tersebut, kemudian dibagi dengan rentang nilai atau nilai maksimum dikurangi nilai minimum dari fitur tersebut.
![Alt text](https://ilmudatapy.com/wp-content/uploads/2020/07/normalisasi-3.png "Title")

### Train-Test-Split
Pada Tahap ini membagi data train dan data test. Data test diset sebesar 20% dari total data.

## Modeling
Pada Tahapan ini digunakan model neural network. Karena data yang digunakan merupakan data time-series maka digunakan  LSTM (Long Short Term Memory) untuk pemodel data time series. Digunakan LSTM karena mampu mengatasi ketergantungan jangka panjang (long term dependencies) pada masukannya.

Terdapat tahapan penyesuaian parameter sebagai berikut:
- Menentukan MAE yang dijadikan patokan akurasi dari model untuk dilakukan evaluasi
- Menggunakan 1 buah layer LSTM seperti berikut
    ```
    model = tf.keras.models.Sequential([
      tf.keras.layers.LSTM(128, return_sequences=True),
      tf.keras.layers.Dropout(0.6),
      tf.keras.layers.Dense(32, activation="relu"),
      tf.keras.layers.Dense(1, activation="relu"),
    ])
    ```
- Membuat fungsi callback untuk menghentikan proses pembelajaran model ketika sudah sesuai dengan target MAE yang ditentukan sebelumnya
    ```
    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        threshold_mae = (df['total_cust'].max() - df['total_cust'].min()) * 20/100
        if(logs.get('val_mae')<threshold_mae):
          print("\nMAE < 20%")
          self.model.stop_training = True
    callbacks = myCallback()
    ```

- Eksekusi model sebagai berikut
    ```
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0000e-03, momentum=0.98)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])
    history = model.fit(train_set,epochs=30, validation_data=(test_set), callbacks=[callbacks])
    ```
    

## Evaluation
Metrik yang digunakan pada variabel kontinyu ini digunakan metrik MAE (Margin Absolute Error).
MAE mengukur besarnya rata-rata kesalahan dalam serangkaian prediksi, tanpa mempertimbangkan arahnya. Ini adalah rata-rata sampel uji dari perbedaan absolut antara prediksi dan pengamatan aktual di mana semua perbedaan individu memiliki bobot yang sama.

![Alt text](https://www.gstatic.com/education/formulas2/472522532/en/mean_absolute_error.svg "Title")

$$MAE$$ = Mean Absolute Error
$$yi$$ = nilai prediksi
$$xi$$ = nilai asli
$$n$$ = jumlah data

Pada proyek ini diset MAE sebesar 20%, ketika MAE pada validation berada dibawah 20% maka model dianggap sudah sesuai dengan target. Ketika nilai MAE pada validation diatas 20% maka akan terus dilakukan pembelajaran sebanyak epoch yang telah di set.
Disini didapat threshold mae senilai 0.2. Danhasil validation mae dihentikan di nilai 0.1950.

![Alt text](https://raw.githubusercontent.com/yunusffz/ml-terapan-proyek-1/main/images/threshold-mae.PNG "Title")
![Alt text](https://raw.githubusercontent.com/yunusffz/ml-terapan-proyek-1/main/images/evaluasi-model.PNG "Title")


## Referensi
VE Sathishkumar, J Park, Y Cho.
        &emsp; &emsp;Using data mining techniques for bike sharing demand prediction in metropolitan city
         &emsp; &emsp;Computer Communications (2020)
