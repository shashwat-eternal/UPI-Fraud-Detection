# UPI Fraud Detection — Resume & Interview Material

## Resume Bullets (pick 2-3 depending on space)

**Option A — technical depth:**
Built an end-to-end UPI fraud detection system (Python, scikit-learn, FastAPI) comparing Decision Tree, Random Forest, and KNN classifiers; engineered an unsupervised Isolation Forest anomaly score and behavioral risk features that lifted Random Forest test accuracy to 99.4% (F1 0.962), a 13-point improvement over the reference paper's baseline feature set.

**Option B — outcome-focused, shorter:**
Designed and deployed a machine learning fraud-detection pipeline (150K transactions, 61 engineered features) achieving 99.4% test accuracy with Random Forest; served predictions via a FastAPI REST backend with automated test coverage (17 tests, 100% passing).

**Option C — leads with rigor:**
Implemented a leakage-free ML pipeline (stratified train/test split before SMOTE balancing, cross-validated hyperparameter tuning) for real-time UPI fraud classification; built and containerized a FastAPI inference service consumed by a browser-based demo dashboard.

## Interview Talking Points (What → My Role → Challenge → Outcome)

**What:**
A machine learning system that classifies UPI transactions as fraudulent or legitimate in real time, extending a published research paper's approach (Random Forest, KNN, Decision Tree on account/transaction-ID/amount) with a much richer engineered feature set and a deployed API.

**My Role:**
End-to-end individual project — dataset design, exploratory analysis, preprocessing, feature engineering, model training/tuning for all three classifiers, evaluation, REST API deployment, frontend demo, and automated testing.

**Challenge:**
The base paper's models topped out around 87% accuracy using only account name, transaction ID, and amount. The core challenge was: what additional signal could be engineered from a transaction that a bank realistically has access to, without needing per-user transaction history? I addressed this by engineering time-based (cyclical hour encoding), behavioral (risk-flag aggregates, velocity ratios), and — most impactfully — an **unsupervised Isolation Forest anomaly score**, fit only on training data without ever seeing the fraud label, which alone correlated at 0.65 with the actual fraud outcome.

A second challenge was avoiding a data leakage mistake that's easy to make with imbalanced fraud data: applying SMOTE *before* the train/test split, which lets synthetic points derived from the same original transaction leak into both sets and inflates test accuracy artificially. I deliberately split first, balanced only the training set, and kept the test set at its real ~8% fraud rate — worth mentioning proactively if asked about methodology, since it shows awareness of a common pitfall.

**Outcome:**
All three models improved substantially over the paper's reported numbers — Random Forest went from 86.67% to 99.39% accuracy (F1 0.962), Decision Tree from 78.67% to 99.14%, KNN from 72.05% to 97.88%. Deployed the winning model behind a FastAPI backend with a working demo UI, and backed the whole pipeline with a 17-test automated suite.

## Likely Follow-Up Questions to Prepare For

- **"Why did Random Forest outperform Decision Tree?"** → Ensembling many trees on bootstrapped samples with random feature subsets reduces variance/overfitting that a single tree is prone to — consistent with both this project's results and the base paper's own conclusion.
- **"Why does KNN have lower precision?"** → It labels based on the majority class among nearest neighbors in feature space; near the fraud/legitimate decision boundary, that boundary is less sharp than a tree's axis-aligned splits, so more legitimate transactions near risky-looking behavior get misclassified as fraud.
- **"Isn't 99% accuracy suspicious / possible overfitting?"** → Be ready to explain the anomaly-score feature is fit unsupervised on training data only, evaluation is on an untouched test set at the real class ratio, and hyperparameters were chosen by cross-validation, not by peeking at the test set — all real safeguards, not just claims.
- **"What would you do with real production data?"** → Point to the Future Enhancements section: real transaction data validation, per-account transaction history features, deep learning sequence models, and streaming inference.
- **"Why synthetic data?"** → Real UPI logs are regulated/private (NPCI, banks); be upfront about this rather than implying the data is real.
