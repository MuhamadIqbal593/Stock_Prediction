# Import library
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Init time start and end 
start = "2017-01-01"
today = date.today().strftime("%Y-%m-%d")

# Judul aplikasi
st.title("Aplikasi Prediksi Saham")

# Tambahkan symbol Saham
saham = ("BBCA.JK", "BYAN.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "TPIA.JK", "UNVR.JK", "BBNI.JK", "ICBP.JK")

# Tombol pilih
pil_saham = st.selectbox("Pilih Saham", saham)

# Pilih rentang bulan
n_month = st.slider("Jumlah Bulan Yang Akan Diprediksi: ", 1, 12) # 1-12 bulan
period = n_month * 30

# Download dataset dari yahoo finance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    # Masukan format tanggal 
    data.reset_index(inplace=True)
    return data

# Masukkan fungsi load data
data_load_state = st.text("Memuat Data...")
data = load_data(pil_saham)
data_load_state.text("Data Berhasil Dimuat!")

# Tampilkan raw datasets
st.subheader("Data Mentah")
st.write(data.tail()) # Tampilkan data harga terbaru

# Buat fungsi plot data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'harga_pembukaan'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'harga_penutupan'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Tampilkan fungsi plot raw data_load
plot_raw_data()

# Buat fungsi pelatihan dengan fbprophet
df_train = data[['Date', 'Close']] 
df_train = df_train.rename(columns={'Date':'ds', 'Close':'y'})

# Init prophet 
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period) # tampilkan sesuai pilihan tahun
forecast = m.predict(future)

# Tampilkan dalam bentuk diagram 
st.subheader('Forecast Data')
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Tampilkan dalam bentuk diagram 
st.subheader('Komponen Prediksi')
fig2 = m.plot_components(forecast)
st.write(fig2)