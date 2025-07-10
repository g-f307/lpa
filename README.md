
# 🌦️ Análise Climática Avançada

Aplicação interativa desenvolvida com **Streamlit** que permite **visualizar, comparar e prever dados climáticos** de diferentes cidades utilizando a API do [OpenWeatherMap](https://openweathermap.org/api).

Além da visualização de gráficos, o sistema realiza **previsões de temperatura** com base em regressão linear, oferecendo também estatísticas detalhadas e comparação entre cidades.

---

## 📌 FUNCIONALIDADES

- Busca de dados climáticos de qualquer cidade do mundo.
- Visualização gráfica (temperatura, umidade ou vento) para 24h, 3 ou 5 dias.
- Previsão automática de temperatura usando modelo de **regressão linear**.
- Exibição de estatísticas resumidas (média, máxima e mínima).
- Comparação lado a lado entre duas cidades.
- Tabela com dados brutos processados.
- Cache de resultados para otimizar o uso da API.

---

## 🧠 VISÃO GERAL TÉCNICA

### Fluxo de Funcionamento:

1. **Entrada do Usuário**: O usuário escolhe uma cidade e um período (24h, 3 ou 5 dias).
2. **Busca de Dados**: A aplicação consulta a API OpenWeatherMap e armazena os dados em cache.
3. **Pré-processamento**:
   - Conversão de datas e horas
   - Extração de variáveis como hora, dia da semana, mês, etc.
4. **Treinamento do Modelo**:
   - Um modelo de regressão linear é treinado com os dados disponíveis para prever a próxima temperatura.
5. **Visualização**:
   - Gráficos interativos com Altair para temperatura, umidade e vento
   - Comparação entre duas cidades (opcional)
   - Estatísticas descritivas (média, mín., máx.)
   - Previsão para as próximas 3 horas
6. **Exibição de dados brutos filtráveis**.

---

## 📁 ESTRUTURA DO CÓDIGO

| Seção | Função Principal | Descrição |
|-------|------------------|-----------|
| `buscar_dados_climaticos` | Requisição + pré-processamento | Consulta API e transforma resposta JSON em DataFrame |
| `prever_temperatura` | Modelo de Regressão Linear | Treina e prevê temperatura futura |
| Sidebar | Entrada do usuário | Seleção de cidade, período e variável |
| Gráficos | Altair charts | Linhas para temperatura, umidade ou vento |
| Estatísticas | Métricas calculadas | Média, máxima, mínima |
| Previsão | Resultado do modelo | Temperatura prevista para 3h à frente |
| Dados brutos | Tabela filtrável | Exibição dos dados originais tratados |

---

## 📈 VARIÁVEIS USADAS NA REGRESSÃO:

| Variável | Significado |
|----------|-------------|
| `hour` | Hora do dia |
| `day_of_week` | Dia da semana (0 = Segunda) |
| `day_of_year` | Dia do ano (1 a 365/366) |
| `month` | Mês (1 a 12) |
| `time_since_start_hours` | Horas desde o primeiro dado |
| `temp`, `humidity`, `wind_speed` | Valores atuais de clima |

---

## 🛠️ TECNOLOGIAS UTILIZADAS:
  
| Tecnologia         | Finalidade                           |
|--------------------|--------------------------------------|
| Python             | Lógica da aplicação                  |
| Streamlit          | Interface Web                        |
| Pandas             | Manipulação de dados                 |
| Scikit-Learn       | Regressão linear e métricas          |
| Altair             | Visualização de gráficos interativos |
| OpenWeatherMap API | Fonte dos dados climáticos           |
| Requests           | Requisições HTTP                     |

---

## 📦 COMO EXECUTAR LOCALMENTE

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

## 🧠 MODELO DE PREVISÃO:

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

## 💡 SUGESTÕES DE CUSTOMIZAÇÕES (PASSÍVEIS DE MELHORIAS)

- Trocar o modelo de regressão por **Random Forest**, **XGBoost** ou **LSTM**.
- Adicionar suporte a múltiplas previsões sequenciais (ex: 6h, 9h à frente).
- Salvar os dados históricos localmente para reuso offline.
- Incluir mapas com geolocalização e temperatura por região.

---

## 📚 REFERÊNCIAS

- [Documentação OpenWeatherMap](https://openweathermap.org/forecast5)
- [Documentação Streamlit](https://docs.streamlit.io/)
- [Altair Charts](https://altair-viz.github.io/)
- [Scikit-Learn Regressão](https://scikit-learn.org/stable/modules/linear_model.html)

---

## 📄 LICENÇA

Este projeto é livre para uso acadêmico e educacional. Para uso comercial, entre em contato com o autor.
