import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


st.title("Previsão de Avaliação de Jogos")
st.subheader("Lucas de Oliveira Assmann")
st.text("2° ano de TADS")


df = "games.csv"
games_df = pd.read_csv(df)

# Filtrar apenas jogos "Indie"
indies = games_df[games_df['Genres'].str.contains('Indie', case=False, na=False)]

# Limpar dados
indies.dropna(subset=['Rating', 'Team'], inplace=True)

# Converter colunas numéricas
def convert_numeric(col):
    return indies[col].replace(r'K', '', regex=True).replace(r',', '', regex=True).astype(float) * 1000

col_num = ['Times Listed', 'Number of Reviews', 'Plays', 'Playing', 'Backlogs', 'Wishlist']
for col in col_num:
    indies[col] = convert_numeric(col)

# Definir variáveis preditoras e alvo
X = indies[['Number of Reviews', 'Plays', 'Wishlist']]
y = indies['Rating']

# Dividir os dados em treino e teste
trX, tsX, trY, tsY = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar para navegação entre páginas
page = st.sidebar.selectbox("Escolha o modelo de predição", ["Regressão Linear", "Random Forest"])

# Função para exibir o modelo de Regressão Linear
def linear_regression():
    model = LinearRegression()
    model.fit(trX, trY)

    # Fazer predições
    predic = model.predict(tsX)

    # Avaliar o modelo
    mse = mean_squared_error(tsY, predic)
    r2 = r2_score(tsY, predic)

    # Exibir resultados no Streamlit
    st.header("Resultados com Regressão Linear")
    st.write(f"Erro Médio Quadrático (MSE): {mse:.2f}")
    st.write(f"Coeficiente de Determinação (R²): {r2:.2%}")

    # Exibir gráfico de comparação
    st.subheader("Comparação das Previsões")
    fig, aux = plt.subplots(figsize=(10, 6))
    aux.scatter(tsY, predic, color="blue")
    aux.plot(tsY, tsY, color="red", linestyle="--", label="Linha Ideal")
    aux.set_xlabel("Avaliações Reais")
    aux.set_ylabel("Avaliações Previstas")
    aux.set_title("Previsão de Avaliações - Regressão Linear")
    aux.legend()
    st.pyplot(fig)

# Função com Random Forest
def random_forest():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(trX, trY)

    # Fazer predições
    predic = model.predict(tsX)

    # Avaliar o modelo
    mse = mean_squared_error(tsY, predic)
    r2 = r2_score(tsY, predic)

    # Exibir resultados no Streamlit
    st.header("Resultados com Random Forest")
    st.write(f"Erro Médio Quadrático (MSE): {mse:.2f}")
    st.write(f"Coeficiente de Determinação (R²): {r2:.2%}")

    # Exibir gráfico de comparação
    st.subheader("Comparação das Previsões")
    fig, aux = plt.subplots(figsize=(10, 6))
    aux.scatter(tsY, predic, color="green")
    aux.plot(tsY, tsY, color="red", linestyle="--", label="Linha Ideal")
    aux.set_xlabel("Avaliações Reais")
    aux.set_ylabel("Avaliações Previstas")
    aux.set_title("Previsão de Avaliações - Random Forest")
    aux.legend()
    st.pyplot(fig)

# Exibir a página conforme a escolha do usuário
if page == "Regressão Linear":
    linear_regression()
elif page == "Random Forest":
    random_forest()
