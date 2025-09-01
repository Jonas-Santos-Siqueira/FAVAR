# A Factor-Augmented Vector Autoregressive (FAVAR)

**FAVAR (Bernanke, Boivin & Eliasz, 2005)** em Python com API inspirada no `statsmodels`.

> Este pacote implementa o procedimento **em dois passos** do BBE (2005):  
> 1) Extração de fatores por **PCA** a partir de um painel informacional grande `X` (padronizado).  
> 2) **Rotação/limpeza** dos fatores via regressão dos PCs sobre `[PCs_slow, R_t]` e **remoção** do componente contemporâneo do instrumento de política `R_t` (ex.: FFR), usando apenas variáveis *slow-moving* para ancorar a identificação.  
> 3) Estimação de um **VAR** em `[F̂_t, Y_t]` com identificação recursiva (Cholesky) e o instrumento **ordenado por último**.  
> 4) Projeção de IRFs para **qualquer série observável** `X_j` via a equação de medida \(X_t \approx \Lambda F_t + \Gamma Y_t\).

## Instalação (local)
```bash
pip install -e .
```
ou simplesmente copie `src/favar_bbe` para seu projeto e importe `favar_bbe`.

## Dependências
- `numpy`, `pandas`, `scipy`, `statsmodels`

## Estrutura mínima dos dados
- `X`: `DataFrame` **T x N** de séries macro (recomendado padronizar — o `fit()` faz isso se `standardize=True`).
- `Y`: `DataFrame` **T x M** de observáveis do VAR (ex.: `["IP", "CPI", "FFR"]`), contendo `policy_var`.

Opcional:
- `slow_columns`: lista com as colunas de `X` consideradas **slow-moving** (preços/quantidades). Caso não forneça, o modelo usa todas as colunas de `X` como *slow* por conveniência.

## Uso rápido
```python
import pandas as pd
from favar_bbe import FAVARModel

# X: DataFrame T x N
# Y: DataFrame T x M (deve conter a coluna "FFR", por exemplo)
slow_cols = [...]  # sua lista de slow-moving (opcional, mas recomendado)

model = FAVARModel(K=3, p=13, policy_var="FFR", slow_columns=slow_cols, standardize=True)
res = model.fit(X, Y)

print(res.summary())                 # resumo do ajuste e do VAR
irfs_Y = model.impulse_response(60)  # IRFs para Y (ex.: IP, CPI, FFR)
fc_Y   = model.forecast(12)          # previsões de Y

# IRFs projetadas para QUALQUER série X (em unidades originais)
irfs_X = model.impulse_response_X(horizon=60, scale="original")
# ou para um subconjunto específico:
irfs_prices = model.impulse_response_X(
    horizon=60, X_cols=["CPI_core","PPI_total","Oil_price"], scale="original"
)
```

## Exemplo completo (sintético)
Veja `examples/synthetic_demo.py` para um exemplo auto-contido que:
1. Gera um painel `X` e um `Y` sintéticos,
2. Ajusta o FAVAR,
3. Plota algumas IRFs (requer `matplotlib`).

## Design de API
- Classe principal: **`FAVARModel`**
  - `fit(X, Y) -> FAVARResults`
  - `forecast(steps)`
  - `impulse_response(horizon, shock=policy_var)`
  - `impulse_response_X(horizon, shock=policy_var, X_cols=None, scale="original"|"std")`
  - `summary()`

### Sobre a rotação/limpeza (BBE, Seção III)
Após a PCA, os PCs de `X` estimam combinações lineares do espaço \((F_t, Y_t)\).  
O BBE remove o componente contemporâneo de `R_t` **apenas dos PCs** via regressão dos PCs de `X` sobre `[PCs_slow, R_t]` e subtração do termo proporcional a `R_t`. Isso evita que a identificação do choque de política seja contaminada por efeitos contemporâneos de `R_t` nos fatores.

### Identificação do choque de política
O VAR é identificado por **Cholesky**, com o instrumento de política **ordenado por último**. A hipótese é que fatores (derivados de *slow-moving*) **não reagem dentro do mês** a choques de política.

### Mapeando IRFs para o espaço X
A equação de medida é estimada por MQO:
\[ X_t \approx [\Lambda\ \Gamma]\,[\hat F_t, Y_t]^\top. \]
Dadas as IRFs de \([\hat F, Y]\), projetamos de volta para cada série `X_j`:
\[ \text{IRF}_{X}(h) = [\hat\Lambda\ \hat\Gamma]\,\text{IRF}_{[\hat F,Y]}(h). \]
Se `standardize=True`, as respostas são reescaladas para unidades originais multiplicando pelo desvio-padrão de cada série.

## Referência
- Bernanke, B., Boivin, J., & Eliasz, P. (2005). *Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach*. Quarterly Journal of Economics.

## Licença
MIT


## Exemplo com dados reais (template)

Coloque seus arquivos em `data/`:
- `data/X_panel.csv` — painel informacional grande (T x N). Inclua uma coluna `date` (YYYY-MM) **ou** deixe a primeira coluna como índice temporal.
- `data/Y_macro.csv` — observáveis do VAR (T x M), incluindo a série de política (ex.: `FFR`).
- (Opcional) `data/slow_columns.txt` — nomes (um por linha) das séries de `X` consideradas *slow-moving*.

Depois rode:
```bash
python examples/real_data_template.py
```
Os resultados vão para `outputs/` (IRFs e previsões em CSV; gráficos se `matplotlib` estiver disponível).

---

## Exemplo com dados reais (via JSON)

Há um exemplo pronto em `examples/real_data_example.py` que lê as configurações de `examples/real_config.json`.

### Passos
1. Coloque seus arquivos `X.csv` e `Y.csv` na pasta `examples/` com a primeira coluna `date` (YYYY-MM/AAAA-MM ou YYYY-MM-DD) e demais colunas como séries.
2. Edite `examples/real_config.json`:
   - `X_path`, `Y_path`: caminhos dos CSVs
   - `policy_var`: nome da variável de política em `Y`
   - `K`, `p` (ou `select_order: true` para escolher por AIC)
   - `slow_columns`: lista de colunas ou o caminho para um CSV (`slow_columns.csv`) com a coluna `name`
   - `irf_horizon`, `X_irf_cols`, `X_irf_scale`
   - `forecast_steps`
3. Execute:
```bash
python examples/real_data_example.py
```
4. Saídas (`examples/output`):
   - `irf_Y.csv` — IRFs para as variáveis de `Y`
   - `irf_X.csv` — IRFs projetadas para as séries `X` (todas ou subconjunto)
   - `forecast_Y.csv` — previsões para `Y`

> Dica: se `select_order` for `true`, o script escolhe `p` pelo AIC usando `statsmodels`.
