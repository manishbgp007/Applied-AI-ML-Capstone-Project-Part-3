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
