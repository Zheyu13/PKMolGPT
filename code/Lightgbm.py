import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm
import lightgbm as lgb
import joblib
from lightgbm import early_stopping, log_evaluation
import warnings
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Disable all warnings and RDKit logging
warnings.filterwarnings("ignore")
logging.getLogger('rdkit').setLevel(logging.ERROR)

# Disable RDKit deprecation warnings
import os
os.environ['RDKIT_DISABLE_DEPRECATION_WARNINGS'] = '1'

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ========= 0. Set random seed ==========
SEED = 42
np.random.seed(SEED)

# ========= Constants for fingerprint sizes (minimal edits) ==========
ECFP_DIM = 2048
MACCS_DIM = 166   # use 166 as the intended MACCS feature count
RDKIT_FP_DIM = 2048

def _ensure_length(arr, length):
    """Ensure fingerprint array has exact `length` by trimming or padding zeros."""
    arr = np.asarray(arr, dtype=int).ravel()
    if arr.size > length:
        return arr[:length]
    elif arr.size < length:
        pad = np.zeros(length - arr.size, dtype=int)
        return np.concatenate([arr, pad])
    else:
        return arr

# ========= 1. Load data ==========
print("Loading data...")
df = pd.read_csv("Np.csv")
df_train = df[df["split"] == "train"]
df_val   = df[df["split"] == "val"]
df_test  = df[df["split"] == "test"]

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# ========= 2. Enhanced feature generation functions ==========

# Initialize fingerprint generators
try:
    # Use new RDKit API with MorganGenerator and explicit sizes
    ecfp4_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=ECFP_DIM, includeChirality=True
    )
    maccs_generator = rdFingerprintGenerator.GetMACCSKeysGenerator()
    # Explicitly request RDKit fpSize
    rdkit_generator = rdFingerprintGenerator.GetRDKitFingerprintGenerator(fpSize=RDKIT_FP_DIM)
    print("Using new RDKit fingerprint generators")
except AttributeError:
    # Fallback to older RDKit API
    from rdkit.Chem import AllChem
    ecfp4_generator = None
    maccs_generator = None
    rdkit_generator = None
    print("Using legacy RDKit fingerprint methods")

def get_polyketide_specific_features(smiles):
    """Extract polyketide-specific molecular features"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(15)
    
    try:
        features = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.TPSA(mol),
            Descriptors.FractionCsp3(mol),
            # Polyketide-specific features
            len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=O'))),  # Carbonyl count
            len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),  # Hydroxyl count
            len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O'))),  # Ester count
            len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)N'))),  # Amide count
            len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=C'))),  # Double bond count
            len(mol.GetSubstructMatches(Chem.MolFromSmarts('C#C'))),  # Triple bond count
        ]
        return np.array(features)
    except:
        return np.zeros(15)

def get_multiple_fingerprints(smiles):
    """Generate multiple molecular fingerprints (ECFP4 + MACCS + RDKit) with enforced lengths"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(ECFP_DIM + MACCS_DIM + RDKIT_FP_DIM)
    
    try:
        if ecfp4_generator is not None:
            # Use newer RDKit API
            ecfp4 = np.array(ecfp4_generator.GetFingerprint(mol))
            maccs = np.array(maccs_generator.GetFingerprint(mol))
            rdkit_fp = np.array(rdkit_generator.GetFingerprint(mol))
        else:
            # Use older RDKit API
            from rdkit.Chem import AllChem
            ecfp4 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP_DIM))
            # MACCS keys may have a different length depending on RDKit; ensure length
            try:
                maccs = np.array(AllChem.GetMACCSKeysFingerprint(mol))
            except Exception:
                # fallback to building MACCS via rdFingerprintGenerator if available or zeros
                maccs = np.zeros(MACCS_DIM, dtype=int)
            rdkit_fp = np.array(AllChem.RDKFingerprint(mol, fpSize=RDKIT_FP_DIM))
        
        # Ensure lengths match expected dims
        ecfp4 = _ensure_length(ecfp4, ECFP_DIM)
        maccs = _ensure_length(maccs, MACCS_DIM)
        rdkit_fp = _ensure_length(rdkit_fp, RDKIT_FP_DIM)
        
        # Combine fingerprints
        combined = np.concatenate([ecfp4, maccs, rdkit_fp])
        return combined
    except Exception as e:
        print(f"Warning: Failed to generate fingerprints for {smiles}: {e}")
        return np.zeros(ECFP_DIM + MACCS_DIM + RDKIT_FP_DIM)

def augment_smiles(smiles, n_augment=3):
    """Generate SMILES augmentations"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    
    augmented = [smiles]
    for _ in range(n_augment):
        try:
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            if rand_smiles not in augmented:
                augmented.append(rand_smiles)
        except:
            continue
    
    return augmented

# ========= 3. Generate enhanced feature matrix ==========
print("Generating enhanced features...")

# Generate fingerprints
tqdm.pandas(desc="Generating fingerprints")
X_train_fp = np.vstack(df_train["smiles"].progress_apply(get_multiple_fingerprints))
X_val_fp   = np.vstack(df_val["smiles"].progress_apply(get_multiple_fingerprints))
X_test_fp  = np.vstack(df_test["smiles"].progress_apply(get_multiple_fingerprints))

# Generate polyketide-specific features
tqdm.pandas(desc="Generating polyketide features")
X_train_pk = np.vstack(df_train["smiles"].progress_apply(get_polyketide_specific_features))
X_val_pk   = np.vstack(df_val["smiles"].progress_apply(get_polyketide_specific_features))
X_test_pk  = np.vstack(df_test["smiles"].progress_apply(get_polyketide_specific_features))

# Combine features
X_train = np.hstack([X_train_fp, X_train_pk])
X_val   = np.hstack([X_val_fp, X_val_pk])
X_test  = np.hstack([X_test_fp, X_test_pk])

print(f"Feature matrix shape: {X_train.shape}")

y_train = df_train["label"].values
y_val   = df_val["label"].values
y_test  = df_test["label"].values

# ========= 4. Data augmentation for training set ==========
print("Performing data augmentation...")
augmented_smiles = []
augmented_labels = []

for idx, row in df_train.iterrows():
    smiles = row["smiles"]
    label = row["label"]
    
    # Add original sample
    augmented_smiles.append(smiles)
    augmented_labels.append(label)
    
    # Add augmented samples (only for positive samples to balance)
    if label == 1:
        aug_smiles_list = augment_smiles(smiles, n_augment=2)
        for aug_smiles in aug_smiles_list[1:]:  # Skip original
            augmented_smiles.append(aug_smiles)
            augmented_labels.append(label)

# Generate features for augmented data
print("Generating features for augmented data...")
aug_df = pd.DataFrame({
    "smiles": augmented_smiles,
    "label": augmented_labels
})

X_train_aug_fp = np.vstack(aug_df["smiles"].progress_apply(get_multiple_fingerprints))
X_train_aug_pk = np.vstack(aug_df["smiles"].progress_apply(get_polyketide_specific_features))
X_train_aug = np.hstack([X_train_aug_fp, X_train_aug_pk])
y_train_aug = np.array(augmented_labels)

print(f"Augmented training set: {len(y_train_aug)} samples")

# ========= 5. Train optimized LightGBM model ==========
print("Training optimized LightGBM classifier...")

# Optimized hyperparameters
model = lgb.LGBMClassifier(
    objective="binary",
    num_leaves=128,
    learning_rate=0.03,
    n_estimators=1000,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=20,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=SEED,
    verbose=-1
)

# Train with augmented data
model.fit(
    X_train_aug,
    y_train_aug,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[
        early_stopping(stopping_rounds=50, first_metric_only=True),
        log_evaluation(period=100),
    ],
)

# ========= 6. Cross-validation ==========
print("Performing cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Use a fresh (unfitted) model for cross_val_score to avoid leakage
cv_model = lgb.LGBMClassifier(**model.get_params())
cv_scores = cross_val_score(cv_model, X_train_aug, y_train_aug, cv=cv, scoring='roc_auc')
print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ========= 7. Model evaluation ==========
print("Evaluating model...")
y_val_pred  = model.predict_proba(X_val)[:, 1]
y_test_pred = model.predict_proba(X_test)[:, 1]

# ROC-AUC scores
print("\nROC-AUC Scores:")
print(f"Val  AUC:  {roc_auc_score(y_val,  y_val_pred):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_test_pred):.4f}")

# Find optimal threshold on validation set
prec, rec, thr = precision_recall_curve(y_val, y_val_pred)
f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_thresh = thr[np.argmax(f1_scores)]
print(f"\nBest threshold on Val (F1-opt): {best_thresh:.4f}")

# Classification report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, (y_test_pred > best_thresh), digits=4))

# ========= 8. Feature importance analysis ==========
print("Analyzing feature importance...")
feature_names = (
    [f"ECFP4_{i}" for i in range(ECFP_DIM)] +
    [f"MACCS_{i}" for i in range(MACCS_DIM)] +
    [f"RDKit_{i}" for i in range(RDKIT_FP_DIM)] +
    ["MolWt", "LogP", "HBD", "HBA", "RotatableBonds", "RingCount", 
     "AromaticRings", "TPSA", "FractionCsp3", "CarbonylCount", 
     "HydroxylCount", "EsterCount", "AmideCount", "DoubleBondCount", "TripleBondCount"]
)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 important features:")
print(feature_importance.head(20))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
sns.barplot(data=top_features, x='importance', y='feature')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# ========= 9. Save model and results ==========
print("Saving model and results...")
joblib.dump(model, "polyketide_lgb_model.pkl")

# Save feature importance
feature_importance.to_csv("feature_importance.csv", index=False)

# Save predictions
results_df = pd.DataFrame({
    'smiles': df_test['smiles'],
    'true_label': y_test,
    'predicted_prob': y_test_pred,
    'predicted_label': (y_test_pred > best_thresh).astype(int)
})
results_df.to_csv("prediction_results.csv", index=False)

print("Model saved as polyketide_lgb_model.pkl")
print("Feature importance saved as feature_importance.csv")
print("Prediction results saved as prediction_results.csv")
print("Feature importance plot saved as feature_importance.png")

# ========= 10. Summary statistics ==========
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Training samples: {len(y_train_aug)}")
print(f"Validation samples: {len(y_val)}")
print(f"Test samples: {len(y_test)}")
print(f"Feature dimensions: {X_train.shape[1]}")
print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Validation AUC: {roc_auc_score(y_val, y_val_pred):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_test_pred):.4f}")
print(f"Optimal threshold: {best_thresh:.4f}")
print("="*50)

