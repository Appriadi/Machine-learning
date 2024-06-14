import pandas as pd
from data_loader import load_data
from model import train_model, predict, evaluate_model

def main():
    # Path ke file SALES.txt
    data_path = '../data/SALES.txt'

    # Memuat data
    data = load_data(data_path)

    # Menampilkan data
    print("Data Sales:")
    print(data.head())
    print()

    # Memisahkan variabel independen (X) dan dependen (y)
    X = data[['Sales']]
    y = data['Advertising']

    # Melatih model
    print("Training model...")
    model = train_model(X, y)

    # Membuat prediksi untuk nilai Sales yang spesifik
    sales_to_predict = [[50], [100], [150]]
    predictions = predict(model, sales_to_predict)

    # Menampilkan hasil prediksi
    results = pd.DataFrame({'Sales (million $)': [50, 100, 150], 'Predicted Advertising Cost (million $)': predictions})
    print("\nPrediksi biaya Advertising:")
    print(results)
    print()

    # Evaluasi model
    y_pred = model.predict(X)
    rmse, r2 = evaluate_model(y, y_pred)

    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R2 Score): {r2:.4f}')

if __name__ == "__main__":
    main()
