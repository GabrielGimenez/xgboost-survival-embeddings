# %%
from xgbse import XGBSETimeVariantBCE
from xgbse.converters import convert_to_structured
from pycox.datasets import metabric
import numpy as np
from sklearn.model_selection import train_test_split


# %%
# getting data
df = metabric.read_df()

df.head()

# %%
# splitting to X, T, E format
X = df.drop(["duration", "event"], axis=1)
T = df["duration"]
E = df["event"]
y = convert_to_structured(T, E)

# splitting between train, and validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=0
)
TIME_BINS = np.arange(15, 315, 15)
TIME_BINS

# %%

var_features = np.random.rand(len(df), len(TIME_BINS))
# %%
xgbse_model = XGBSETimeVariantBCE(n_jobs=1)
xgbse_model.fit(
    X_train,
    y_train,
    num_boost_round=50,
    time_bins=TIME_BINS,
    var_features=var_features[: len(X_train)],
)

# %%
