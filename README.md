# Membuka dan menampilkan data
- Mendownload data dari URL : url = "https://www.econometrics.com/intro/SALES.txt"
                                     data = pd.read_table(url, sep='\s+')
- Menampilkan semua baris data : print(data)
- Menampilkan informasi kolom : print(data.columns)
# Pembuatan Model Regresi
- Memisahkan variabel independen (X) dan dependen (y)
X = data[['Sales']]
y = data['Advertising']

- Membuat model regresi linear
model = LinearRegression()
model.fit(X, y)

- Membuat prediksi untuk nilai Sales yang spesifik
sales_to_predict = [[50], [100], [150]]
predictions = model.predict(sales_to_predict)

- Menampilkan hasil prediksi
results = pd.DataFrame({'Sales (million $)':  [50, 100, 150], 'Predicted Advertising Cost (million $)': predictions})
print(results)
# Evaluasi Model
- Prediksi menggunakan data training
y_pred = model.predict(X)
- Menghitung RMSE dan R2 score
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)
