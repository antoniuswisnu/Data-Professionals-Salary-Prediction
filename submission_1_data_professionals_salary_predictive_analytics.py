# -*- coding: utf-8 -*-
"""Submission 1_Data Professionals Salary Predictive Analytics

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1m6GycyVxPnaBrDQczvw6h0MJ2luH2Lb2

# Import Libraries

Tahap pertama adalah mengimpor semua pustaka (libraries) yang diperlukan untuk proyek ini. Pustaka yang diimpor meliputi:
- `numpy` untuk operasi numerik.
- `matplotlib.pyplot` dan `seaborn` untuk visualisasi data.
- `pandas` untuk manipulasi dan analisis data, terutama untuk bekerja dengan DataFrame.
- `%matplotlib inline` adalah magic command di Jupyter untuk memastikan plot ditampilkan langsung di bawah sel kode.
- `kagglehub` untuk mengunduh dataset langsung dari Kaggle.
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

"""# Data Loading

**Proses:**<br>
Pada tahap ini, kita memuat dataset yang akan digunakan.
- `kagglehub.dataset_download()` digunakan untuk mengunduh dataset "Analytics Industry Salaries 2022 India" dari Kaggle.
- `pd.read_csv()` digunakan untuk membaca file `Partially Cleaned Salary Dataset.csv` dari path yang telah diunduh dan menyimpannya ke dalam DataFrame pandas yang bernama `df`.
"""

import kagglehub

path = kagglehub.dataset_download("iamsouravbanerjee/analytics-industry-salaries-2022-india")

print("Path to dataset files:", path)

url = '/root/.cache/kagglehub/datasets/iamsouravbanerjee/analytics-industry-salaries-2022-india/versions/14/Partially Cleaned Salary Dataset.csv'
df = pd.read_csv(url)
df

"""**Hasil:**<br>
Dataset berhasil dimuat ke dalam DataFrame `df`. Dari output, kita dapat melihat 5 baris pertama dan 5 baris terakhir dari dataset, yang terdiri dari 4339 baris dan 6 kolom. Kolom-kolom tersebut adalah `Unnamed: 0`, `Company Name`, `Job Title`, `Salaries Reported`, `Location`, dan `Salary`.

# EDA

**Proses:**<br>
Tahap ini adalah bagian dari *Exploratory Data Analysis* (EDA), di mana kita mulai memahami karakteristik dasar dari dataset.
- `df.info()` digunakan untuk mendapatkan ringkasan singkat tentang DataFrame, termasuk tipe data setiap kolom dan jumlah nilai non-null.
- `df.describe()` digunakan untuk melihat statistik deskriptif dasar untuk kolom-kolom numerik (seperti mean, standar deviasi, min, max, dan kuartil).

## Deskripsi Variable
"""

df.info()

df.describe()

"""**Hasil:**<br>
- **df.info()**: Hasilnya menunjukkan bahwa tidak ada nilai yang hilang (missing values) di setiap kolom. Tipe data kolom sudah sesuai, dengan `object` untuk data kategorikal dan `int64`/`float64` untuk data numerik.
- **df.describe()**: Dari output ini, kita dapat melihat bahwa kolom `Salary` memiliki rentang nilai yang sangat besar dan standar deviasi yang tinggi, yang mengindikasikan kemungkinan adanya *outliers* (pencilan).

## Menangani Missing Value dan Outlier

**Proses:**<br>
Pada tahap ini, kita fokus pada pembersihan data dari nilai yang tidak valid dan *outliers*.
- Pertama, kita memeriksa apakah ada nilai 0 di kolom `Salaries Reported` dan `Salary`, karena nilai 0 bisa jadi merupakan data yang tidak valid.
- Selanjutnya, kita membuat visualisasi `boxplot` untuk kolom `Salaries Reported` dan `Salary` untuk mengidentifikasi adanya *outliers* secara visual.
- Terakhir, kita menggunakan metode IQR (*Interquartile Range*) untuk menghapus *outliers*. Data yang berada di luar rentang (Q1 - 1.5*IQR) dan (Q3 + 1.5*IQR) akan dihapus.
"""

salaries_reported = (df['Salaries Reported'] == 0).sum()
salary = (df.Salary == 0).sum()

print("Nilai 0 di kolom Salaries Reported ada: ", salaries_reported)
print("Nilai 0 di kolom Salary ada: ", salary)

df.loc[(df['Salary'] == 0)]

df = df.loc[(df[['Unnamed: 0']]!=0).all(axis=1)]

df.shape

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Salary'], kde=True)
plt.title('Distribusi Salary Sebelum Transformasi')

df['Salary_Log'] = np.log1p(df['Salary'])  # log1p = log(1+x) untuk menghindari log(0)

plt.subplot(1, 2, 2)
sns.histplot(df['Salary_Log'], kde=True)
plt.title('Distribusi Salary Setelah Log Transformation')
plt.tight_layout()

sns.boxplot(x=df['Salary'])

sns.boxplot(x=df['Salaries Reported'])

numeric_col = ['Salaries Reported', 'Salary']

df[numeric_col]

Q1 = df[numeric_col].quantile(0.10)
Q3 = df[numeric_col].quantile(0.90)

IQR = Q3-Q1
df = df[~((df[numeric_col]<(Q1-2.0*IQR))|(df[numeric_col]>(Q3+2.0*IQR))).any(axis=1)]

df.shape

"""**Hasil:**<br>
- Tidak ditemukan nilai 0 pada kolom target.
- Boxplot dengan jelas menunjukkan adanya banyak *outliers* pada kedua kolom numerik, terutama pada nilai-nilai yang sangat tinggi.
- Setelah penghapusan *outliers* dengan metode IQR, jumlah baris data berkurang dari 4339 menjadi 3801. Ini menunjukkan bahwa sejumlah besar data pencilan telah berhasil dihilangkan, membuat dataset lebih representatif untuk pemodelan.

## Univariate Analysis

**Proses:**<br>
Tahap ini melibatkan analisis terhadap masing-masing variabel (fitur) secara individual untuk memahami distribusinya.
- **Fitur Kategorikal**: Kita melakukan analisis pada fitur `Company Name` dan `Job Title`. Kita menghitung frekuensi kemunculan setiap nilai unik (`value_counts()`) dan memvisualisasikannya dalam bentuk bar chart untuk melihat kategori mana yang paling umum.
- **Fitur Numerik**: Kita menggunakan `df.hist()` untuk membuat histogram dari semua fitur numerik. Ini membantu kita memahami distribusi dari data numerik setelah proses pembersihan *outliers*.
"""

numerical_features = ['Salaries Reported', 'Salary']
categorical_features = ['Company Name', 'Job Title', 'Location']

"""### Categorical Features

#### Fitur Cut
"""

feature = categorical_features[0]
count = df[feature].value_counts().nlargest(10)
percent = 100*df[feature].value_counts(normalize=True)
df_company_name = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_company_name)
count.plot(kind='bar', title=feature);

"""#### Fitur Color"""

feature = categorical_features[1]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_job_title = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_job_title)
count.plot(kind='bar', title=feature);

"""### Numerical Features"""

df.hist(bins=50, figsize=(20,15))
plt.show()

"""**Hasil:**<br>
- **Company Name**: Analisis menunjukkan bahwa fitur ini memiliki kardinalitas yang sangat tinggi (banyak sekali nama perusahaan unik), sehingga bar chart tidak terlalu informatif.
- **Job Title**: Distribusinya lebih jelas, di mana "Data Scientist", "Data Analyst", dan "Data Engineer" adalah tiga jabatan yang paling sering muncul dalam dataset.
- **Fitur Numerik**: Histogram menunjukkan distribusi dari `Salaries Reported` dan `Salary`. Fitur `Salary` terlihat lebih mendekati distribusi normal setelah *outliers* dihapus, meskipun masih sedikit miring ke kanan (*right-skewed*).

## Multivariate Analysis

**Proses:**<br>
Di tahap ini, kita menganalisis hubungan antara dua atau lebih variabel.
- **Categorical Features**: Kode ini bertujuan untuk membuat `catplot` yang menunjukkan rata-rata `Salary` untuk setiap kategori pada fitur `Company Name`, `Job Title`, dan `Location`. Namun, kode ini menghasilkan `NameError`, yang perlu diperbaiki. Tujuannya adalah untuk melihat bagaimana fitur-fitur ini mempengaruhi gaji.
- **Numerical Features**: Kita menggunakan `sns.pairplot()` untuk melihat hubungan antar semua fitur numerik dan `sns.heatmap()` untuk memvisualisasikan matriks korelasi.

### Categorical Features
"""

cat_features = df.select_dtypes(include='object').columns.to_list()

for col in cat_features:
    top_5_categories = df[col].value_counts().nlargest(5).index.tolist()
    df_filtered = df[df[col].isin(top_5_categories)].copy()
    sns.catplot(x=col, y="Salary", kind="bar", dodge=False, height = 4, aspect = 3,  data=df_filtered, palette="Set3")
    plt.title("Rata-rata 'Salary' Relatif terhadap Top 5 - {}".format(col))
    plt.show()

"""### Numerical Features"""

sns.pairplot(df, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

df.drop(['Unnamed: 0'], inplace=True, axis=1)
df.head()

"""**Hasil:**<br>
- **Pairplot & Heatmap**: Hasilnya menunjukkan korelasi antara `Salaries Reported` dan `Salary`. Terdapat korelasi positif yang sangat lemah (0.16), yang mengindikasikan bahwa jumlah laporan gaji yang lebih banyak cenderung sedikit berhubungan dengan gaji yang sedikit lebih tinggi, namun hubungannya tidak kuat.

# Data Preparation

**Proses:**<br>
Tahap ini mempersiapkan data agar siap digunakan untuk melatih model machine learning.
- **Encoding Fitur Kategori**: Fitur kategorikal seperti `Company Name`, `Job Title`, dan `Location` diubah menjadi format numerik menggunakan *One-Hot Encoding* (`pd.get_dummies`). Metode ini membuat kolom biner baru untuk setiap nilai unik pada fitur kategorikal.
- **Reduksi Dimensi dengan PCA**: *Principal Component Analysis* (PCA) diterapkan pada fitur `Salaries Reported` untuk mengurangi dimensinya menjadi satu komponen utama. Ini dilakukan untuk menyederhanakan fitur tanpa kehilangan banyak informasi variansnya.
- **Train-Test Split**: Dataset dibagi menjadi data latih (*train*) dan data uji (*test*) dengan perbandingan 90:10. Ini penting agar model dapat dievaluasi pada data yang belum pernah "dilihat" sebelumnya.
- **Standarisasi**: Fitur numerik hasil PCA (`dimension`) kemudian distandarisasi menggunakan `StandardScaler`. Proses ini mengubah skala fitur sehingga memiliki rata-rata 0 dan standar deviasi 1. Scaler di-*fit* hanya pada data latih untuk menghindari kebocoran data (*data leakage*).

## Encoding Fitur Kategori
"""

def create_target_encoding(df, categorical_col, target_col):
    encoding_map = df.groupby(categorical_col)[target_col].mean().to_dict()
    new_col_name = f'{categorical_col}_target_enc'
    df[new_col_name] = df[categorical_col].map(encoding_map)
    return df, new_col_name

df, company_target_enc = create_target_encoding(df, 'Company Name', 'Salary')

from sklearn.preprocessing import  OneHotEncoder
df = pd.concat([df, pd.get_dummies(df['Company Name'], prefix='Company Name')],axis=1)
df = pd.concat([df, pd.get_dummies(df['Job Title'], prefix='Job Title')],axis=1)
df = pd.concat([df, pd.get_dummies(df['Location'], prefix='Location')],axis=1)

df['Salary_per_Report'] = df['Salary'] / (df['Salaries Reported'] + 1)
df.drop(['Company Name','Job Title','Location'], axis=1, inplace=True)
df.head()

sns.pairplot(df[['Salaries Reported', 'Salary']], plot_kws={"s": 2});

from sklearn.decomposition import PCA

pca = PCA(n_components=1, random_state=123)
pca.fit(df[['Salaries Reported']])
princ_comp = pca.transform(df[['Salaries Reported']])

pca.explained_variance_ratio_.round(3)

from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123)
pca.fit(df[['Salaries Reported']])
df['dimension'] = pca.transform(df[['Salaries Reported']]).flatten()
df.drop(['Salaries Reported'], axis=1, inplace=True)

"""## Train-Test-Split"""

df.head()

from sklearn.model_selection import train_test_split

X = df.drop(['Salary', 'Salary_Log'],axis = 1)
y = df['Salary']
y_log = df['Salary_Log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)
_, _, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.1, random_state=123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""## Standarisasi"""

from sklearn.preprocessing import StandardScaler

numerical_features = ['dimension', 'Salary_per_Report', company_target_enc]
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train[numerical_features].describe().round(4)

"""**Hasil:**<br>
- Data kategorikal telah berhasil diubah menjadi numerik, meskipun ini menambah jumlah kolom secara signifikan.
- Fitur `Salaries Reported` telah direduksi menjadi satu fitur bernama `dimension`.
- Dataset telah terbagi menjadi `X_train`, `X_test`, `y_train`, dan `y_test`.
- Fitur numerik pada data latih telah distandarisasi dan siap untuk pemodelan.

# Model Development

**Proses:** Kita memilih dan menyiapkan tiga model regresi yang berbeda untuk memprediksi gaji:
1.  **K-Nearest Neighbors (KNN):** Model non-parametrik yang memprediksi berdasarkan 'kedekatan' dengan data latih.
2.  **Random Forest:** Model *ensemble* yang membangun banyak *decision tree* dan menggabungkan hasilnya untuk prediksi yang lebih stabil dan akurat.
3.  **AdaBoost (Adaptive Boosting):** Model *ensemble* yang secara sekuensial melatih model-model lemah dan memberikan bobot lebih pada data yang salah diklasifikasikan pada iterasi sebelumnya.

Model-model ini disimpan dalam sebuah dictionary untuk kemudahan iterasi, kemudian masing-masing model dilatih menggunakan data `X_train` dan `y_train_log` (target yang sudah ditransformasi log) dengan memanggil metode `.fit()`.

## K-Nearest Neighbor
"""

models = pd.DataFrame(index=['train_mse', 'test_mse', 'train_r2', 'test_r2'],
                      columns=['KNN', 'KNN_log', 'RandomForest', 'RandomForest_log', 'Boosting', 'Boosting_log'])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
              'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5,
                       scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)
knn = knn_grid.best_estimator_
print(f"Best KNN parameters for original target: {knn_grid.best_params_}")

knn = KNeighborsRegressor(n_neighbors=3, weights='uniform')
knn.fit(X_train, y_train)

models.loc['train_mse','KNN'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
models.loc['train_r2','KNN'] = r2_score(y_pred = knn.predict(X_train), y_true=y_train)

knn_log_grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5,
                           scoring='neg_mean_squared_error')
knn_log_grid.fit(X_train, y_log_train)
knn_log = knn_log_grid.best_estimator_
print(f"Best KNN parameters for log target: {knn_log_grid.best_params_}")

knn_log = KNeighborsRegressor(n_neighbors=3, weights='uniform')
knn_log.fit(X_train, y_log_train)
models.loc['train_mse','KNN_log'] = mean_squared_error(y_pred = knn_log.predict(X_train), y_true=y_log_train)
models.loc['train_r2','KNN_log'] = r2_score(y_pred = knn_log.predict(X_train), y_true=y_log_train)

"""## Random Forest"""

from sklearn.ensemble import RandomForestRegressor

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=55), param_grid_rf, cv=5,
                      scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid.fit(X_train, y_train)
RF = rf_grid.best_estimator_
print(f"Best RF parameters for original target: {rf_grid.best_params_}")

RF = RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_split=2, random_state=42)
RF.fit(X_train, y_train)
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
models.loc['train_r2','RandomForest'] = r2_score(y_pred=RF.predict(X_train), y_true=y_train)

rf_log_grid = GridSearchCV(RandomForestRegressor(random_state=55), param_grid_rf, cv=5,
                          scoring='neg_mean_squared_error', n_jobs=-1)
rf_log_grid.fit(X_train, y_log_train)
RF_log = rf_log_grid.best_estimator_
print(f"Best RF parameters for log target: {rf_log_grid.best_params_}")

RF_log = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=2, random_state=42)
RF_log.fit(X_train, y_log_train)
models.loc['train_mse','RandomForest_log'] = mean_squared_error(y_pred=RF_log.predict(X_train), y_true=y_log_train)
models.loc['train_r2','RandomForest_log'] = r2_score(y_pred=RF_log.predict(X_train), y_true=y_log_train)

"""## AdaBoostRegressor"""

from sklearn.ensemble import AdaBoostRegressor

param_grid_boost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'loss': ['linear', 'square', 'exponential']
}

boost_grid = GridSearchCV(AdaBoostRegressor(random_state=55), param_grid_boost, cv=5,
                         scoring='neg_mean_squared_error')
boost_grid.fit(X_train, y_train)
boosting = boost_grid.best_estimator_
print(f"Best AdaBoost parameters for original target: {boost_grid.best_params_}")

from sklearn.ensemble import AdaBoostRegressor
boosting = AdaBoostRegressor(n_estimators=200, learning_rate=0.1, loss='exponential', random_state=42)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
models.loc['train_r2','Boosting'] = r2_score(y_pred=boosting.predict(X_train), y_true=y_train)

boost_log_grid = GridSearchCV(AdaBoostRegressor(random_state=55), param_grid_boost, cv=5,
                             scoring='neg_mean_squared_error')
boost_log_grid.fit(X_train, y_log_train)
boosting_log = boost_log_grid.best_estimator_
print(f"Best AdaBoost parameters for log target: {boost_log_grid.best_params_}")

boosting_log = AdaBoostRegressor(n_estimators=200, learning_rate=0.1, loss='exponential', random_state=42)
boosting_log.fit(X_train, y_log_train)
models.loc['train_mse','Boosting_log'] = mean_squared_error(y_pred=boosting_log.predict(X_train), y_true=y_log_train)
models.loc['train_r2','Boosting_log'] = r2_score(y_pred=boosting_log.predict(X_train), y_true=y_log_train)

"""**Hasil:**<br>
Ketiga model berhasil dilatih. *Mean Squared Error* (MSE) pada data latih dihitung untuk setiap model sebagai metrik performa awal.

# Evaluation Model

**Proses:** Setelah pelatihan, kinerja setiap model dievaluasi menggunakan metrik **Mean Absolute Error (MAE)**. MAE mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual. Prediksi dibuat pada data latih dan data uji, dan MAE dihitung untuk keduanya.
"""

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

models.loc['test_mse','KNN'] = mean_squared_error(y_true=y_test, y_pred=knn.predict(X_test))
models.loc['test_r2','KNN'] = r2_score(y_true=y_test, y_pred=knn.predict(X_test))

models.loc['test_mse','KNN_log'] = mean_squared_error(y_true=y_log_test, y_pred=knn_log.predict(X_test))
models.loc['test_r2','KNN_log'] = r2_score(y_true=y_log_test, y_pred=knn_log.predict(X_test))

models.loc['test_mse','RandomForest'] = mean_squared_error(y_true=y_test, y_pred=RF.predict(X_test))
models.loc['test_r2','RandomForest'] = r2_score(y_true=y_test, y_pred=RF.predict(X_test))

models.loc['test_mse','RandomForest_log'] = mean_squared_error(y_true=y_log_test, y_pred=RF_log.predict(X_test))
models.loc['test_r2','RandomForest_log'] = r2_score(y_true=y_log_test, y_pred=RF_log.predict(X_test))

models.loc['test_mse','Boosting'] = mean_squared_error(y_true=y_test, y_pred=boosting.predict(X_test))
models.loc['test_r2','Boosting'] = r2_score(y_true=y_test, y_pred=boosting.predict(X_test))

models.loc['test_mse','Boosting_log'] = mean_squared_error(y_true=y_log_test, y_pred=boosting_log.predict(X_test))
models.loc['test_r2','Boosting_log'] = r2_score(y_true=y_log_test, y_pred=boosting_log.predict(X_test))

def back_transform_log(log_predictions):
    return np.expm1(log_predictions)

model_dict_log = {'KNN_log': knn_log, 'RandomForest_log': RF_log, 'Boosting_log': boosting_log}
model_dict_original = {'KNN': knn, 'RandomForest': RF, 'Boosting': boosting}

compare_df = pd.DataFrame(index=['Original Scale MSE', 'Original Scale R2'],
                         columns=['KNN', 'KNN_log_back', 'RandomForest', 'RandomForest_log_back',
                                 'Boosting', 'Boosting_log_back'])

for name, model in model_dict_original.items():
    y_pred = model.predict(X_test)
    compare_df.loc['Original Scale MSE', name] = mean_squared_error(y_test, y_pred)
    compare_df.loc['Original Scale R2', name] = r2_score(y_test, y_pred)

for name, model in model_dict_log.items():
    y_log_pred = model.predict(X_test)
    y_pred_back = back_transform_log(y_log_pred)
    compare_df.loc['Original Scale MSE', f"{name.split('_')[0]}_log_back"] = mean_squared_error(y_test, y_pred_back)
    compare_df.loc['Original Scale R2', f"{name.split('_')[0]}_log_back"] = r2_score(y_test, y_pred_back)

plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
compare_df.loc['Original Scale MSE'].plot(kind='bar')
plt.title('Mean Squared Error (MSE) - Lower is Better')
plt.ylabel('MSE')
plt.grid(axis='y')

plt.subplot(2, 1, 2)
compare_df.loc['Original Scale R2'].plot(kind='bar')
plt.title('R² Score - Higher is Better')
plt.ylabel('R² Score')
plt.grid(axis='y')

plt.tight_layout()

feature_importances = pd.DataFrame(
    RF.feature_importances_,
    index=X_train.columns,
    columns=['importance']
).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
feature_importances.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()

print("\nPerbandingan Semua Model:")
print(compare_df)

best_model_name = compare_df.loc['Original Scale R2'].idxmax()
print(f"\nModel terbaik berdasarkan R² score: {best_model_name}")
print(f"R² score: {compare_df.loc['Original Scale R2', best_model_name]:.4f}")
print(f"MSE: {compare_df.loc['Original Scale MSE', best_model_name]:.4f}")

"""**Proses:** Kita membuat scatter plot untuk membandingkan nilai gaji aktual (`y_test`) dengan nilai gaji yang diprediksi oleh model terbaik (AdaBoost). Garis diagonal merah (y=x) ditambahkan sebagai referensi; jika prediksi sempurna, semua titik akan berada di garis ini."""

if '_log_back' in best_model_name:
    model_name = best_model_name.split('_')[0]
    model = model_dict_log[f"{model_name}_log"]
    y_log_pred = model.predict(X_test)
    y_pred = back_transform_log(y_log_pred)
else:
    model = model_dict_original[best_model_name]
    y_pred = model.predict(X_test)

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title(f'Actual vs Predicted Salary - {best_model_name}')
plt.grid(True)

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)

"""**Hasil/Insight:** Plot menunjukkan bahwa sebagian besar titik data berkumpul di sekitar garis diagonal, yang mengindikasikan bahwa prediksi model cukup akurat. Meskipun ada beberapa prediksi yang meleset cukup jauh (outlier), secara umum model mampu menangkap tren dan pola dalam data gaji dengan baik. Ini secara visual mengonfirmasi bahwa model AdaBoost adalah pilihan yang solid untuk tugas prediksi ini.

"""

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict_original.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

for name, model in model_dict_log.items():
    y_log_pred = model.predict(prediksi)
    pred_dict['prediksi_'+name+'_back'] = back_transform_log(y_log_pred).round(1)

pd.DataFrame(pred_dict)

"""**Hasil:**<br>
- Terakhir, kita melakukan prediksi pada satu sampel data uji untuk melihat perbandingan nilai prediksi dari setiap model dengan nilai sebenarnya (`y_true`). Hasil menunjukkan bahwa prediksi RF lah yang paling mendekati nilai (`y_true`)
"""
