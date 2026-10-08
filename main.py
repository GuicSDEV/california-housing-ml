from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """Treina uma árvore com o número de folhas dado e retorna o MAE na validação."""
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)


# 1. Carregar e explorar os dados
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.describe())

# 2. Separar features (X) e target (y)
features = ["MedInc", "HouseAge", "AveRooms", "Latitude", "Longitude"]
X = df[features]
y = df["MedHouseVal"]

# 3. Dividir em treino (80%) e validação (20%)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. Árvore de decisão: buscar o melhor max_leaf_nodes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
best_tree_mae = float("inf")
best_tree_size = None

print("\n--- Árvore de decisão ---")
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"Folhas: {max_leaf_nodes:<5} MAE: {mae:.4f}")
    if mae < best_tree_mae:
        best_tree_mae = mae
        best_tree_size = max_leaf_nodes

print(f"Melhor árvore: {best_tree_size} folhas")

# 5. Random Forest com parâmetros padrão
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(train_X, train_y)
rf_preds_val = rf_model.predict(val_X)
rf_mae = mean_absolute_error(val_y, rf_preds_val)

# 6. Comparar os modelos (target em centenas de milhares de dólares)
print("\n--- Resultado ---")
print(f"Árvore de decisão: MAE {best_tree_mae:.4f} (~${best_tree_mae * 100_000:,.0f})")
print(f"Random Forest:     MAE {rf_mae:.4f} (~${rf_mae * 100_000:,.0f})")

if rf_mae < best_tree_mae:
    melhoria = (best_tree_mae - rf_mae) / best_tree_mae * 100
    print(f"\nA Random Forest errou {melhoria:.1f}% menos que a melhor árvore.")
else:
    print("\nA árvore de decisão teve um erro menor que a Random Forest.")
    