# %%
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("dataset/kaggle/cardio_train.csv", delimiter=';', index_col=0)
df.head()

# %%

# Split into a "World" pool (Attacker has some data, Hospital has some)
# Target Data: The exact rows used to train the Victim Model
# Population Data: Other rows from the same distribution (used for shadow models)
X_train_target, X_population, y_train_target, y_population = train_test_split(
    df.drop('cardio', axis=1), df['cardio'], test_size=0.66, random_state=42
)
X_population, X_holdout, y_population, y_holdout = train_test_split(
    X_population, y_population, test_size=0.5, random_state=42
)
print(f"Target Train Size: {X_train_target.shape[0]}")
print(f"Population Size: {X_population.shape[0]}")
print(f"Holdout Size: {X_holdout.shape[0]}")

# %%

# ==========================================
# 2. TRAIN TARGET MODEL (The Victim)
# ==========================================
# Using Random Forest as it is common in tabular healthcare data
from xgboost import XGBClassifier

target_model = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.5, min_child_weight=1, eval_metric='logloss', random_state=42)
_target_model = clone(target_model)
target_model.fit(X_train_target, y_train_target)

print(f"Target Model Accuracy: {target_model.score(X_train_target, y_train_target):.2f}")
print(f"Test Model Accuracy: {target_model.score(X_holdout, y_holdout):.2f}")

# %%

# ==========================================
# 3. LiRA IMPLEMENTATION
# ==========================================

class LiRAAttack:
    def __init__(self, target_model, n_shadow_models=16):
        self.target_model = target_model
        self.n_shadow_models = n_shadow_models
        self.shadow_models = []
        self.shadow_data_indices = [] # Track which rows were used in which model
        
    def _logit_scale(self, probs):
        """
        Transform probabilities (0-1) to Logit space (-inf to +inf).
        This makes the distribution closer to Gaussian.
        Added epsilon for numerical stability.
        """
        epsilon = 1e-5
        probs = np.clip(probs, epsilon, 1 - epsilon)
        return np.log(probs / (1 - probs))

    def train_shadow_models(self, X_pool, y_pool):
        """
        Train N shadow models on random subsets of the available population data.
        LiRA requires overlapping subsets to get both 'IN' and 'OUT' scores for specific points.
        """
        print(f"Training {self.n_shadow_models} Shadow Models...")
        pool_indices = np.arange(len(X_pool))
        
        for i in range(self.n_shadow_models):
            # Sample 50% of the pool for this shadow model
            sample_indices = np.random.choice(pool_indices, 
                                              size=int(len(X_pool)*0.5), 
                                              replace=False)
            
            # Create mask for tracking
            mask = np.zeros(len(X_pool), dtype=bool)
            mask[sample_indices] = True
            self.shadow_data_indices.append(mask)
            
            # Train Shadow Model
            X_shadow = X_pool.iloc[sample_indices]
            y_shadow = y_pool.iloc[sample_indices]
            
            shadow_model = clone(self.target_model)
            shadow_model.fit(X_shadow, y_shadow)
            self.shadow_models.append(shadow_model)
            
            if (i+1) % 5 == 0: print(f"  - Trained {i+1}/{self.n_shadow_models}")

    def calculate_lira_scores(self, X_target, y_target, X_pool, y_pool):
        """
        The Core LiRA Logic:
        1. Query all shadow models with the Target Record.
        2. Split scores into:
           - IN_scores: Models that DID train on this record (from pool).
           - OUT_scores: Models that DID NOT train on this record.
        3. Fit Gaussian (Mean, Std) to OUT scores.
        4. Calculate Likelihood of the Target Model's score under that Gaussian.
        """
        
        # 1. Get the Target Model's confidence on the target record
        # We only care about the probability of the *True Class*
        target_probs = self.target_model.predict_proba(X_target)
        # Extract prob of correct class
        target_confs = np.array([probs[y] for probs, y in zip(target_probs, y_target)])
        target_logits = self._logit_scale(target_confs)
        
        lira_scores = []
        
        # We iterate through every record we want to attack
        # Note: In a real attack, we need to ensure we have Shadow Models 
        # that effectively simulate "IN" and "OUT" for the specific record.
        # For this demo, we approximate by querying the Shadow Models trained on the Pool.
        
        print("Calculating Likelihood Ratios...")
        
        # Pre-calculate all shadow model predictions for the target set
        # Shape: (n_shadows, n_samples)
        all_shadow_confs = []
        for model in self.shadow_models:
            probs = model.predict_proba(X_target)
            # Get confidence for the true label class
            confs = [
                probs_row[y_true] 
                for probs_row, y_true in zip(probs, y_target)
            ]
            all_shadow_confs.append(self._logit_scale(np.array(confs)))
            
        all_shadow_confs = np.array(all_shadow_confs)
        
        # For each target record
        for i in range(len(X_target)):
            # Get the distribution of scores from shadow models
            # Since we don't know if the target record was in the shadow train sets
            # (because X_target is separate from X_pool in this split scenario),
            # we treat ALL shadow models as "OUT" models (simulating Non-Members).
            # This is the standard "Reference Model" approach when you can't inject the target into shadows.
            
            # If we had trained shadows ON X_target, we would split them here.
            # Current assumption: Null Hypothesis (H_out) = The record is NOT in training.
            
            out_scores = all_shadow_confs[:, i]
            
            # Fit Gaussian to the OUT distribution
            mu_out = np.mean(out_scores)
            std_out = np.std(out_scores) + 1e-8 # Avoid div/0
            
            # Calculate Likelihood of the Target Model's score
            # Pr(obs | Non-Member)
            pr_out = stats.norm.pdf(target_logits[i], loc=mu_out, scale=std_out)
            
            # Simplified LiRA Score (Log Likelihood Ratio approximation)
            # Since we often lack the "IN" distribution without computationally expensive
            # 'Leave-One-In' training, we often use the Z-Score relative to the OUT distribution
            # as a proxy for the Ratio in simplified implementations.
            # High positive Z-score -> Higher than expected for a non-member -> Likely Member
            lira_score = (target_logits[i] - mu_out) / std_out
            lira_scores.append(lira_score)
            
        return np.array(lira_scores)

# %%

# ==========================================
# 4. EXECUTE ATTACK
# ==========================================

# Initialize Attack
attacker = LiRAAttack(_target_model, n_shadow_models=64)

# Train Shadow Models (using the Population data the attacker has)
attacker.train_shadow_models(X_population, y_population)

# Calculate Scores
member_scores = attacker.calculate_lira_scores(X_train_target, y_train_target, X_population, y_population)
non_member_scores = attacker.calculate_lira_scores(X_holdout, y_holdout, X_population, y_population)

# %%

# ==========================================
# 5. VISUALIZATION & METRICS
# ==========================================
plt.figure(figsize=(10, 6))
sns.kdeplot(member_scores, label='Members (Training Data)', fill=True, color='red')
sns.kdeplot(non_member_scores, label='Non-Members (Holdout)', fill=True, color='blue')
plt.title("LiRA Score Distribution: Members vs Non-Members")
plt.xlabel("LiRA Score (Z-score relative to Shadow Models)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%

# ROC Curve
labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
scores = np.concatenate([member_scores, non_member_scores])

# Plot the ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([1e-4, 1.0])
plt.ylim([1e-4, 1.0])
plt.yscale('log')
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
print(f"LiRA Attack AUC: {roc_auc:.2f}")

# %%
print(f"TPR: {tpr[1]}, FPR: {fpr[1]}")


