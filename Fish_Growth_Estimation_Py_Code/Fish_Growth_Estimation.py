from tkinter import *
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Initialize Tkinter window
main = Tk()
main.title("Deep CNN Model for Fish Growth Estimation from Larvae Statistics")
main.geometry("1000x650")
main.config(bg='skyblue')

# Title label
title = Label(main, text="Fish Growth Estimation from Larvae Statistics", justify='center')
title.grid(column=0, row=0)
font = ('times', 15, 'bold')
title.config(bg='orange', fg='white', font=font, height=3, width=105)
title.place(x=50, y=5)

# Global variables
global df
global x, y, x_train, x_test, y_train, y_test
global scaler, pca, cnn_model  # For consistent scaling, PCA, and CNN prediction

def upload():
    global df
    filename = filedialog.askopenfilename(initialdir="Dataset")
    df = pd.read_csv(filename)
    text.insert(END, f"{filename} Loaded\n\n")
    text.insert(END, f"Shape: {df.shape}\n")
    print("Columns:", df.columns.tolist())  # Debug

def preprocessing():
    global df
    df = df.replace('nd', np.NaN)
    df.dropna(inplace=True)  # Drop rows with NaN values
    categorical_cols = ['cruise_id', 'brief_desc', 'common_name']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    text.insert(END, f"\n\nCategorical columns encoded: {categorical_cols}\n")
    text.insert(END, f"All columns: {df.columns.tolist()}\n")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', fmt=".2f")  # annot=False due to many columns
    plt.title('Correlation Heatmap')
    plt.show()

def splitting():
    global x, y, x_train, x_test, y_train, y_test, scaler, pca
    x = df.drop(['growth'], axis=1)
    y = df['growth']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    n_features = x_train_scaled.shape[1]
    n_samples = x_train_scaled.shape[0]
    n_components = min(n_samples, n_features)  # Dynamic PCA components
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    x_train = x_train_pca
    x_test = x_test_pca
    text.insert(END, '\n\nData splitting complete.\n')
    text.insert(END, f"X-train shape: {x_train.shape}, Y-train shape: {y_train.shape}\n")
    text.insert(END, f"X-test shape: {x_test.shape}, Y-test shape: {y_test.shape}\n")
    print(f"x_train_scaled shape: {x_train_scaled.shape}, n_components: {n_components}")

def performance_metrics(algorithm, predict, testY):
    mse = mean_squared_error(testY, predict)
    mae = mean_absolute_error(testY, predict)
    r2 = r2_score(testY, predict)
    text.insert(END, f'\n\n----------{algorithm}------\n\n')
    text.insert(END, f"{algorithm} Mean Squared Error: {mse:.4f}\n")
    text.insert(END, f"{algorithm} Mean Absolute Error: {mae:.4f}\n")
    text.insert(END, f"{algorithm} R^2 Score: {r2:.4f}\n")
    plt.figure()
    plt.scatter(testY, predict)
    plt.plot([testY.min(), testY.max()], [testY.min(), testY.max()], '--r', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f"{algorithm} Predictions vs True Values")
    plt.show()

def svr_Model():
    global x_train, x_test, y_train, y_test
    svr_model_path = 'model/svr_model.joblib'
    os.makedirs('model', exist_ok=True)
    # Always train a new model to match current data dimensions
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_model.fit(x_train, y_train)
    joblib.dump(svr_model, svr_model_path)  # Save the newly trained model
    y_pred_svr = svr_model.predict(x_test)
    performance_metrics('Support Vector Regressor', y_pred_svr, y_test)

def cnn_Model():
    global x_train, x_test, y_train, y_test, cnn_model
    # Reshape for CNN: (samples, features, channels)
    x_train_reshaped = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test_reshaped = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    # Define CNN model
    cnn_model = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=2, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    # Compile and train
    cnn_model.compile(optimizer='adam', loss='mse')
    cnn_model.fit(x_train_reshaped, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Predict
    y_pred_cnn = cnn_model.predict(x_test_reshaped).flatten()
    performance_metrics('Proposed CNN Regression', y_pred_cnn, y_test)

def prediction():
    global scaler, pca, cnn_model
    path = filedialog.askopenfilename(initialdir="test")
    test = pd.read_csv(path)
    test = test.replace('nd', np.NaN)
    test.dropna(inplace=True)
    test = pd.get_dummies(test, columns=['cruise_id', 'brief_desc', 'common_name'], drop_first=True)
    missing_cols = set(df.columns) - set(test.columns)
    for col in missing_cols:
        test[col] = 0
    test = test[df.columns]  # Align with training columns
    test_x = test.drop(['growth'], axis=1)
    test_x_scaled = scaler.transform(test_x)  # Use training scaler
    test_x_pca = pca.transform(test_x_scaled)  # Use training PCA
    test_x_reshaped = test_x_pca.reshape((test_x_pca.shape[0], test_x_pca.shape[1], 1))
    pred = cnn_model.predict(test_x_reshaped).flatten()  # Use CNN for prediction
    test['Prediction'] = pd.Series(pred)
    text.insert(END, '\n\n-------Prediction-------\n\n')
    text.insert(END, f"{test}\n\n")

# GUI Buttons
font = ('times', 15, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.config(bg='blue', fg='Black', width=14, font=font)
uploadButton.place(x=50, y=100)

preprocessButton = Button(main, text="Pre Processing", command=preprocessing)
preprocessButton.config(bg='blue', fg='Black', width=14, font=font)
preprocessButton.place(x=260, y=100)

splitButton = Button(main, text="Splitting", command=splitting)
splitButton.config(bg='blue', fg='Black', width=14, font=font)
splitButton.place(x=470, y=100)

svrButton = Button(main, text="SVR", command=svr_Model)
svrButton.config(bg='blue', fg='Black', width=14, font=font)
svrButton.place(x=680, y=100)

cnnButton = Button(main, text="Proposed= CNN", command=cnn_Model)
cnnButton.config(bg='blue', fg='Black', width=14, font=font)
cnnButton.place(x=890, y=100)

predictButton = Button(main, text="Predict", command=prediction)
predictButton.config(bg='blue', fg='Black', width=14, font=font)
predictButton.place(x=1100, y=100)

# Text widget with scrollbar
font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=200)
text.config(font=font1)

main.mainloop()