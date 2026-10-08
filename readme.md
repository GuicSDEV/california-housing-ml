# California Housing — Árvore de Decisão vs Random Forest

Projeto de estudo que compara dois modelos de regressão do scikit-learn para prever o valor mediano das casas em regiões da Califórnia. Feito para praticar os conceitos do curso [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) do Kaggle.

## O que o projeto faz

1. Carrega o dataset **California Housing** (incluído no scikit-learn)
2. Separa os dados em treino (80%) e validação (20%)
3. Treina **árvores de decisão** com diferentes valores de `max_leaf_nodes` e escolhe a de menor erro
4. Treina uma **Random Forest** com os parâmetros padrão
5. Compara os dois modelos usando o **MAE** (Mean Absolute Error)

## Dados

Cada linha representa uma região da Califórnia (censo de 1990). Features usadas:

| Feature | Descrição |
|---|---|
| `MedInc` | Renda mediana da região |
| `HouseAge` | Idade mediana das casas |
| `AveRooms` | Média de cômodos por casa |
| `Latitude` | Latitude |
| `Longitude` | Longitude |

**Target:** `MedHouseVal`, o valor mediano das casas, em centenas de milhares de dólares.

## Como rodar

```bash
git clone https://github.com/GuicSDEV/california-housing-ml.git
cd california-housing-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Resultados

| Modelo | MAE | Erro médio aproximado |
|---|---|---|
| Árvore de decisão (melhor: XXX folhas) | X.XXXX | ~$XX,XXX |
| Random Forest (padrão) | X.XXXX | ~$XX,XXX |

## O que aprendi

- **Validação:** avaliar o modelo com os mesmos dados do treino dá uma falsa impressão de acerto. Por isso separei uma parte dos dados só para testar.
- **Underfitting x Overfitting:** árvores com poucas folhas são simples demais e erram muito, e árvores com folhas demais decoram o treino. O melhor fica no meio-termo.
- **Random Forest:** fazer a média de muitas árvores diferentes reduz o overfitting e deu um erro menor que a melhor árvore, sem precisar de ajuste.

## Tecnologias

- Python
- pandas
- scikit-learn