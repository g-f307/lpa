
# 🌦️ Análise Climática Avançada

Aplicação interativa desenvolvida com **Streamlit** que permite **visualizar, comparar e prever dados climáticos** de diferentes cidades utilizando a API do [OpenWeatherMap](https://openweathermap.org/api).

Além da visualização de gráficos, o sistema realiza **previsões de temperatura** com base em regressão linear, oferecendo também estatísticas detalhadas e comparação entre cidades.

---

## 📌 Funcionalidades

- Busca de dados climáticos de qualquer cidade do mundo.
- Visualização gráfica (temperatura, umidade ou vento) para 24h, 3 ou 5 dias.
- Previsão automática de temperatura usando modelo de **regressão linear**.
- Exibição de estatísticas resumidas (média, máxima e mínima).
- Comparação lado a lado entre duas cidades.
- Tabela com dados brutos processados.
- Cache de resultados para otimizar o uso da API.

---

## 🛠️ Tecnologias Utilizadas

<div align=center>
  
| Tecnologia         | Finalidade                           |
|--------------------|--------------------------------------|
| Python             | Lógica da aplicação                  |
| Streamlit          | Interface Web                        |
| Pandas             | Manipulação de dados                 |
| Scikit-Learn       | Regressão linear e métricas          |
| Altair             | Visualização de gráficos interativos |
| OpenWeatherMap API | Fonte dos dados climáticos           |
| Requests           | Requisições HTTP                     |

</div>

---

## 📦 Como Executar Localmente

### Pré-requisitos

- [Anaconda ou Miniconda](https://www.anaconda.com/) (Sugestão de ambiente de desenvolvimento)
- Chave de API do OpenWeatherMap (gratuita em: https://openweathermap.org/api)
- Git (para clonar o repositório)

### 1. Crie um Ambiente Virtual com Conda

Recomenda-se isolar o ambiente para evitar conflitos de dependências:

```bash
conda create -n lpa_clima python=3.9 pandas numpy matplotlib requests streamlit scikit-learn jupyter -y
```

Depois, ative o ambiente:

```bash
conda activate lpa_clima
```

### 2. Clone o Repositório

```bash
git clone https://github.com/g-f307/lpa.git
cd lpa
```

### 3. Configure sua Chave da API

No arquivo `LPA_Final.py`, substitua a linha:

```python
API_KEY = "f323af5a10043e6368d0d173fc7cae17"
```

Pela sua própria chave da API, obtida no site do [OpenWeatherMap](https://openweathermap.org/api).

### 4. Execute a Aplicação

```bash
streamlit run LPA_Final.py
```

Acesse no navegador: [http://localhost:8501](http://localhost:8501)

---

## 🧠 Modelo de Previsão

- **Tipo**: Regressão Linear (Scikit-Learn)
- **Features utilizadas**:
  - Hora do dia
  - Dia da semana
  - Dia do ano
  - Mês
  - Tempo desde o início da série
  - Temperatura atual
  - Umidade
  - Velocidade do vento

---

## 📄 Licença

Este projeto é livre para uso acadêmico e educacional. Para uso comercial, entre em contato com o autor.
