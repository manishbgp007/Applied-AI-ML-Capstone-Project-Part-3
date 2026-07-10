# Applied-AI-ML-Capstone-Project-Part-3

### Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

### Task 1: Decision Tree Baseline
The first classification model developed in this section was an **unconstrained Decision Tree Classifier**. The model was trained using the default parameters provided by Scikit-learn, where **`max_depth=None`**. This means that the tree was allowed to grow until all leaf nodes became pure or no further splits were possible.

* **Training the Decision Tree Classifier**
  * The Decision Tree model was trained using the scaled training dataset. During training, the algorithm recursively selected the best feature and split point at each node based on an impurity criterion (such as Gini Impurity or Entropy). This process continued until the tree reached its maximum possible size because no depth restriction was applied.
  * As a result, the model learned highly detailed decision rules from the training data.

* **Model Performance**
  * After training, the model was evaluated on both the training and testing datasets.
  * The following observations were made:
    * **Training Accuracy:** Very high (often close to 100%).
    * **Testing Accuracy:** Significantly lower than the training accuracy.
  * The large gap between training and testing performance indicated that the model performed extremely well on the data it had already seen but struggled to generalize to unseen examples.

* **Interpretation: Overfitting**
  * The observed performance difference is a classic example of **overfitting**.
  * Overfitting occurs when a machine learning model memorizes the training data instead of learning the underlying patterns. Consequently, the model captures random noise and small fluctuations that are specific to the training dataset, leading to poor performance on new data.
  * Because the Decision Tree was allowed to grow without any restrictions, it created many complex branches that perfectly fit the training observations but failed to generalize effectively.

* **Why Decision Trees Tend to Overfit**
  * Decision Trees are considered **high-variance models** because they are highly sensitive to the training data.
  * The algorithm follows a **greedy approach**, selecting the best split at each node based only on the current data. Once a split is made, it is never reconsidered or revised later in the tree-building process.
  * As the tree continues to grow, it begins to model not only the true patterns in the data but also random noise and minor irregularities. This increases model complexity and makes predictions less reliable on unseen data.

* **Need for Model Regularization**
  * The baseline results demonstrate that an unconstrained Decision Tree is prone to overfitting. To improve generalization, the model complexity must be controlled using hyperparameters such as:
    * **`max_depth`** – Limits the maximum depth of the tree.
    * **`min_samples_split`** – Specifies the minimum number of samples required to split a node.
    * **`min_samples_leaf`** – Ensures that each leaf node contains a minimum number of observations.
    * **`max_leaf_nodes`** – Restricts the total number of leaf nodes.
These parameters reduce the complexity of the tree, helping the model focus on meaningful patterns rather than memorizing the training data.

* **Outcome**
  * The unconstrained Decision Tree served as the baseline classification model for this section. Although it achieved very high training accuracy, the noticeably lower testing accuracy revealed clear evidence of overfitting. This experiment highlighted the importance of controlling tree complexity and motivated the use of pruning and hyperparameter tuning in the subsequent tasks to improve the model's ability to generalize to unseen data.


### Task 2: Controlled Decision Tree
To reduce the overfitting observed in the baseline Decision Tree model, a **controlled (regularized) Decision Tree Classifier** was trained by limiting the complexity of the tree. Hyperparameters were introduced to prevent the model from growing excessively deep and fitting noise in the training data.

* **Training the Controlled Decision Tree**
  * A new Decision Tree Classifier was trained using the following hyperparameters:
    * **`max_depth = 5`**
    * **`min_samples_split = 20`**
These constraints limited the growth of the tree and encouraged the model to learn only the most important patterns from the training data rather than memorizing every observation.

* **Role of Hyperparameters**
  * Maximum Depth (`max_depth = 5`)
  * The **`max_depth`** parameter controls the maximum number of levels that the tree is allowed to grow.
  * By restricting the depth to **5**, the model becomes less complex and avoids creating unnecessary branches that fit random fluctuations in the training data.
  * Effects of limiting tree depth:
    * Reduces model complexity.
    * Decreases variance.
    * Improves generalization to unseen data.
    * Introduces a small amount of bias, which is often acceptable if it reduces overfitting.
  * Minimum Samples Required to Split (`min_samples_split = 20`)
  * The **`min_samples_split`** parameter specifies the minimum number of samples required before a node can be split into child nodes.
  * Setting this value to **20** prevents the algorithm from creating splits based on very small subsets of data.
  * This reduces the likelihood of learning patterns caused by random noise rather than meaningful relationships.
  * Benefits include:
    * Prevents unnecessary splits.
    * Produces a simpler and more stable tree.
    * Reduces sensitivity to noisy observations.
    * Improves the robustness of the model.

* **Model Performance**
  * The controlled Decision Tree was evaluated on both the training and testing datasets.
  * The following observations were made:
    * **Training Accuracy:** Lower than the unconstrained Decision Tree.
    * **Testing Accuracy:** Higher than the unconstrained Decision Tree.
 Although the model did not memorize the training data as effectively, it performed better on unseen data, indicating improved predictive capability.

* **Interpretation
  * The decrease in training accuracy combined with the improvement in testing accuracy indicates that the controlled Decision Tree achieved better **generalization**.
  * This demonstrates the effect of **regularization**, where limiting model complexity helps reduce overfitting.
  * Instead of learning random noise present in the training data, the controlled tree focused on capturing the most significant and meaningful patterns, resulting in more reliable predictions on new observations.
  * This behavior illustrates the **bias–variance trade-off**:
    * Increasing model constraints introduces a small amount of bias.
    * At the same time, it substantially reduces variance.
    * The overall result is improved performance on unseen data.

* **Outcome**
  * The controlled Decision Tree successfully addressed the overfitting problem observed in the baseline model. By applying the hyperparameters **`max_depth = 5`** and **`min_samples_split = 20`**, the model became simpler, more stable, and better at generalizing to unseen data. This experiment demonstrated that carefully tuning tree complexity can significantly improve predictive performance and produce a more robust machine learning model.


### Task 3: Gini vs Entropy**
  * To compare different splitting criteria used by Decision Tree classifiers, two models were trained using different impurity measures:
    * **Gini Impurity (`criterion='gini'`)**
    * **Entropy (`criterion='entropy'`)**
  * Both models were trained using the same training dataset and evaluated on the same test dataset to ensure a fair comparison.
    
* **Gini Impurity**
  * The first Decision Tree was trained using **Gini Impurity** as the splitting criterion.
  * The Gini impurity is calculated as:
    **Gini Impurity = 1 − Σ(pi²)**
  * where **pi** represents the probability of each class within a node.
  * The Gini index measures how often a randomly selected sample would be incorrectly classified if it were labeled according to the class distribution of the node.
  * A **Gini value of 0** indicates that all samples in the node belong to the same class, meaning the node is perfectly pure.
  * Lower Gini values represent purer nodes and better splits.

* **Entropy**
  * The second Decision Tree was trained using **Entropy** as the splitting criterion.
  * Entropy is calculated as:
    **Entropy = − Σ(pi × log₂(pi))**
  * where **pi** is the probability of each class within the node.
  * Entropy measures the amount of uncertainty or disorder present in a node.
    * **Entropy = 0** indicates complete purity, meaning all samples belong to a single class.
    * Higher entropy values indicate greater class mixture and uncertainty.
The Decision Tree algorithm selects the split that produces the largest reduction in entropy, commonly referred to as **Information Gain**.

* **Comparing Gini and Entropy**
  * Both Decision Tree models were evaluated using the same performance metrics on the test dataset.
  * The comparison focused on:
    * Training accuracy
    * Testing accuracy
    * Overall classification performance
The results showed that the **Gini** and **Entropy** criteria produced **very similar test accuracy**, indicating that both methods identified nearly the same decision boundaries for this dataset.

* **Interpretation**
  * Although Gini Impurity and Entropy use different mathematical formulas to measure node impurity, they often produce very similar Decision Trees in practice.
  * The experiment demonstrated that:
    * **Gini Impurity = 0** means the node is perfectly pure because all samples belong to a single class.
    * **Entropy = 0** also indicates complete purity with no uncertainty.
    * Both criteria generally select similar feature splits and achieve comparable predictive performance.
In many real-world machine learning problems, the choice between Gini and Entropy has only a small effect on the final accuracy.

* **Practical Differences**
  * Although their performance is often similar, there are some practical differences:
  * **Gini Impurity**
    * Computationally faster because it does not require logarithmic calculations.
    * Commonly used as the default criterion in Scikit-learn.
    * Preferred when computational efficiency is important.
  * **Entropy**
    * Based on Information Theory.
    * Uses Information Gain to determine the best split.
    * May occasionally produce slightly different tree structures but usually similar predictive performance.

* **Outcome**
  * The comparison between **Gini Impurity** and **Entropy** showed that both splitting criteria produced nearly identical classification performance on the test dataset. This indicates that either criterion can be effectively used for Decision Tree learning, with the choice often depending on computational efficiency or user preference rather than significant differences in predictive accuracy.


### Task 4: Random Forest Classifier
To improve the performance and generalization ability of the Decision Tree model, a **Random Forest Classifier** was implemented. Random Forest is an ensemble learning algorithm that combines the predictions of multiple Decision Trees to produce a more accurate and robust classifier. By averaging the predictions of many trees, Random Forest reduces the risk of overfitting that is commonly observed in a single Decision Tree.

* **Training the Random Forest Model**
  * The Random Forest model was trained using the following hyperparameters:
    * **Number of Trees (`n_estimators`) = 100**
    * **Maximum Tree Depth (`max_depth`) = 10**
  * Each of the 100 decision trees was trained independently, and the final prediction was obtained through **majority voting** among all trees.
  * Limiting the maximum depth to **10** helped prevent individual trees from becoming overly complex while maintaining sufficient flexibility to learn meaningful patterns.

* **Model Evaluation**
  * The trained Random Forest model was evaluated using both the training and testing datasets.
  * The following performance metrics were reported:
    * **Training Accuracy**
    * **Testing Accuracy**
    * **ROC-AUC Score**
  * These metrics were used to evaluate both the learning capability and the generalization performance of the model.
    * **Training Accuracy** indicates how well the model fits the training data.
    * **Testing Accuracy** measures the model's performance on unseen data.
    * **ROC-AUC** evaluates the model's ability to distinguish between the two classes across different decision thresholds.
A high test accuracy and a high ROC-AUC score indicate that the model generalizes well and makes reliable predictions.

* **Feature Importance Analysis**
  * One advantage of Random Forest is its ability to estimate the relative importance of each input feature.
  * The **top five most important features** were identified based on the feature importance scores produced by the model.
  * Feature importance in Random Forest is calculated as the **average reduction in Gini impurity** contributed by a feature across all decision tree splits. These reductions are aggregated over all trees in the forest and then normalized to produce the final importance score.
  * Features with higher importance scores contribute more to the prediction process and have a greater influence on the model's decisions.

* **Understanding the Bagging Technique**
  * Random Forest is based on the **Bootstrap Aggregating (Bagging)** technique.
  * The algorithm improves prediction performance through two sources of randomness:
    * Bootstrap Sampling
      * Each decision tree is trained on a **bootstrap sample**, which is created by randomly sampling observations **with replacement** from the original training dataset.
      * Because sampling is performed with replacement, some observations may appear multiple times in a bootstrap sample, while others may not appear at all.
      * This creates diversity among the individual trees.
    * Random Feature Selection
      * At every split in a decision tree, Random Forest does **not** evaluate all available features.
      * Instead, it randomly selects a subset of features, typically equal to **√(number of features)** for classification tasks, and chooses the best split only from that subset.
      * This additional randomness reduces the similarity between trees and increases the diversity of the ensemble.

* **Why Random Forest Performs Better**
  * A single Decision Tree is considered a **high-variance model** because small changes in the training data can produce very different tree structures.
  * Random Forest reduces this variance by averaging the predictions of many independent trees.
  * As a result:
    * Individual trees may make different errors.
    * Averaging their predictions reduces the impact of those errors.
    * The final model becomes more stable and generalizes better to unseen data.
This ensemble approach usually provides higher predictive accuracy and better robustness than a single deep Decision Tree.

* **Outcome**
  * The Random Forest Classifier successfully combined **100 Decision Trees** to produce a more accurate and reliable classification model. The model was evaluated using **training accuracy, testing accuracy, and ROC-AUC**, while the **top five most important features** provided insights into the variables that most strongly influenced the predictions. The experiment also demonstrated how **bagging**, **bootstrap sampling**, and **random feature selection** reduce variance and improve generalization, making Random Forest one of the most effective ensemble learning algorithms for classification tasks.
