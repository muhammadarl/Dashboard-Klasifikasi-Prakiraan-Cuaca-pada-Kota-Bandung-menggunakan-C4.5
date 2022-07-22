import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.write(
    '''
# Web Apps - Klasifikasi Cuaca di Kota Bandung
Aplikasi Berbasis Web untuk mengklasifikasi **cuaca**
'''
)
img = Image.open('cloudy.png')
st.image(img, use_column_width=False)
st.sidebar.header('Parameter Input')
def input_user():
    Suhu = st.sidebar.slider('Suhu', 16, 25, 31)
    Lembap = st.sidebar.slider('Lembap', 55, 70, 95)
    KecepatanAngin = st.sidebar.slider('Kecepatan Angin', 0, 10, 20)
    arah_angin = st.sidebar.slider('Arah Angin', 1, 4, 8)
    data = {'suhu':Suhu, 'lembap':Lembap, 'kecepatan_angin':KecepatanAngin, 'arah_angin':arah_angin}
    return pd.DataFrame(data=[data], index=[0])
df = input_user()

dataset = pd.read_excel('Dataset_Bandung.xlsx')
st.subheader('Arah Angin')
st.write('''
1. Utara
2. Selatan
3. Timur
4. Barat
5. Timur Laut
6. Barat Daya
7. Tenggara
8. Barat Laut''')
st.subheader('Parameter Input')
st.write(df)
num_var = ['temperatur', 'kecepatan_angin', 'kelembapan']
dataset.drop('tanggal', axis=1, inplace=True)
dataset.drop('waktu', axis=1, inplace=True)
dataset['arah_angin'] = dataset['arah_angin'].replace({'utara': 1, 'selatan': 2, 'timur': 3, 'barat': 4, 'timur laut': 5, 'barat daya': 6, 'tenggara': 7, 'barat laut': 8})
X = dataset.iloc[:, :4]
y = dataset['prakiraan']
model = DecisionTreeClassifier()
model.fit(X, y)
prediksi = model.predict(df)
prediksi_proba = model.predict_proba(df)

st.subheader('Label Kelas')
st.write(dataset['prakiraan'].unique())
st.subheader('Hasil Klasifikasi')
st.write(prediksi)
st.subheader('Hasil Klasifikasi proba')
st.write(prediksi_proba)