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

* **Interpretation**
  * The decrease in training accuracy combined with the improvement in testing accuracy indicates that the controlled Decision Tree achieved better **generalization**.
  * This demonstrates the effect of **regularization**, where limiting model complexity helps reduce overfitting.
  * Instead of learning random noise present in the training data, the controlled tree focused on capturing the most significant and meaningful patterns, resulting in more reliable predictions on new observations.
  * This behavior illustrates the **bias–variance trade-off**:
    * Increasing model constraints introduces a small amount of bias.
    * At the same time, it substantially reduces variance.
    * The overall result is improved performance on unseen data.

* **Outcome**
  * The controlled Decision Tree successfully addressed the overfitting problem observed in the baseline model. By applying the hyperparameters **`max_depth = 5`** and **`min_samples_split = 20`**, the model became simpler, more stable, and better at generalizing to unseen data. This experiment demonstrated that carefully tuning tree complexity can significantly improve predictive performance and produce a more robust machine learning model.


### Task 3: Gini vs Entropy
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


### Task 4a: Gradient Boosting Classifier

To further improve classification performance, a **Gradient Boosting Classifier** was implemented. Unlike Random Forest, which builds many independent decision trees in parallel, Gradient Boosting builds trees **sequentially**, where each new tree learns to correct the prediction errors made by the previous trees. This boosting strategy often results in higher predictive accuracy, especially for complex datasets.

* **Training the Gradient Boosting Model**
  * The Gradient Boosting Classifier was trained using the following hyperparameters:
    * **Number of Estimators (`n_estimators`) = 100**
    * **Learning Rate (`learning_rate`) = 0.1**
    * **Maximum Tree Depth (`max_depth`) = 3**

Each tree in the ensemble was intentionally kept shallow (depth = 3), allowing the model to learn gradually while reducing the risk of overfitting.

* **Model Evaluation**
  * The trained Gradient Boosting model was evaluated on both the training and testing datasets.
  * The following evaluation metrics were reported:
    * **Training Accuracy**
    * **Testing Accuracy**
    * **ROC-AUC Score**
  * These metrics were used to assess both the learning performance and the generalization capability of the model.
    * **Training Accuracy** measures how well the model fits the training data.
    * **Testing Accuracy** evaluates performance on unseen data.
    * **ROC-AUC** measures the model's ability to distinguish between positive and negative classes across all classification thresholds.
A higher ROC-AUC indicates better classification performance.

* **How Gradient Boosting Works**
  * Gradient Boosting is an **ensemble learning algorithm** that combines multiple weak learners (typically shallow decision trees) into a strong predictive model.
  * The algorithm operates sequentially:
    1. The first tree is trained on the original dataset.
    2. The prediction errors (residuals) from the first tree are calculated.
    3. A second tree is trained specifically to correct those errors.
    4. Each subsequent tree continues to reduce the remaining prediction errors.
    5. The final prediction is obtained by combining the predictions of all trees.
Because every new tree focuses on correcting mistakes made by earlier trees, the overall model becomes increasingly accurate.

* **Role of the Hyperparameters**
  * Number of Estimators (`n_estimators = 100`)
  * This parameter specifies the total number of trees in the boosting sequence.
  * Using more trees generally allows the model to learn more complex patterns, although an excessively large number of trees may increase computation time and the risk of overfitting.
  * Learning Rate (`learning_rate = 0.1`)
  * The learning rate determines how much each new tree contributes to the final prediction.
  * **Lower learning rate**
    * Slower learning.
    * Often better generalization.
    * Usually requires more trees.
  * **Higher learning rate**
    * Faster learning.
    * May fit the training data more quickly.
    * Increases the risk of overfitting if set too high.
  * A value of **0.1** provides a good balance between learning speed and model stability.
  * Maximum Tree Depth (`max_depth = 3`)
  * The maximum depth controls the complexity of each individual decision tree.
  * Keeping the trees shallow:
    * Reduces overfitting.
    * Improves generalization.
    * Encourages each tree to learn only simple patterns while the ensemble captures complex relationships.

* **Interpretation**
  * The Gradient Boosting model demonstrated how sequential learning can improve predictive performance.
  * Unlike Random Forest, where trees are built independently, Gradient Boosting continuously improves the model by correcting previous mistakes.
  * This often results in:
    * Higher prediction accuracy.
    * Better ROC-AUC performance.
    * Stronger ability to capture complex relationships in the data.
However, if the **learning rate** is too high or the **tree depth** is too large, the model may begin to memorize the training data, leading to **overfitting**. Therefore, careful tuning of these hyperparameters is essential to achieve a balance between model complexity and generalization.

* **Outcome**
  * The Gradient Boosting Classifier was successfully trained using **100 estimators**, a **learning rate of 0.1**, and **maximum tree depth of 3**. The model was evaluated using **training accuracy, testing accuracy, and ROC-AUC**, demonstrating its effectiveness as a powerful ensemble learning technique. The experiment highlighted the advantages of sequential boosting while emphasizing the importance of selecting appropriate hyperparameters to maximize predictive performance without causing overfitting.


### Task 4b: Feature Ablation
To evaluate the contribution of individual features to the model's predictive performance, a **feature ablation experiment** was performed. The objective was to determine whether the least important features identified by the Random Forest model were actually useful or whether they could be removed without significantly affecting performance.

* **Identifying the Least Important Features**
  * The feature importance scores generated by the **Random Forest Classifier** were examined. These scores indicate how much each feature contributes to reducing impurity and improving the model's predictions.
  * Based on these importance scores, the **five least important features** were identified. These features had the smallest contribution to the overall predictive performance of the model.

* **Creating a Reduced Feature Set**
  * After identifying the least important features, they were removed from the dataset to create a reduced feature matrix.
  * A new Random Forest Classifier was then trained using only the remaining features while keeping the same model hyperparameters as the original model.
  * This ensured that any difference in performance was due only to the removal of the selected features.

* **Comparing Model Performance**
  * The performance of the two Random Forest models was compared:
    * **Full Model:** Trained using all available features.
    * **Reduced Model:** Trained after removing the five least important features.
  * The primary evaluation metric used for comparison was:
    * **ROC-AUC Score**
Comparing the ROC-AUC values allowed the impact of feature removal on classification performance to be measured objectively.

* **Interpretation of Results**
  * The feature ablation results were interpreted as follows:
  * **If the ROC-AUC remained nearly the same or improved:**
    * The removed features contained little useful predictive information.
    * They were likely noisy, redundant, or irrelevant.
    * Removing them simplified the model without reducing prediction quality.
  * **If the ROC-AUC decreased noticeably:**
    * The removed features contributed meaningful predictive information.
    * Their removal reduced the model's ability to distinguish between classes.
    * Those features should be retained in the final model.
This analysis helped determine which features were truly valuable for prediction.

* **Production Trade-Off**
  * Feature selection is not only important for improving model interpretability but also for practical deployment in production systems.
  * Removing uninformative features provides several advantages:
    * Reduces computational cost during prediction.
    * Decreases memory usage.
    * Speeds up inference time.
    * Simplifies model maintenance and deployment.
    * Makes the model easier to interpret.
However, feature removal is only beneficial if the resulting decrease in predictive performance is **minimal or negligible**. A significant drop in ROC-AUC would indicate that the removed features contain important information and should be retained.

* **Outcome**
  * The feature ablation experiment evaluated the impact of removing the **five least important features** identified by the Random Forest model. By comparing the ROC-AUC scores of the full and reduced models, it was possible to determine whether those features were useful or unnecessary. This experiment demonstrated the importance of balancing predictive performance with model simplicity, showing how feature selection can reduce inference cost and improve deployment efficiency while maintaining strong classification performance.


### Task 5: Cross-Validated Model Comparison
To obtain a more reliable estimate of model performance, **5-fold Stratified Cross-Validation** was performed. Instead of evaluating the models using only a single train-test split, cross-validation assesses model performance across multiple subsets of the data, providing a more robust estimate of generalization ability.

* **Cross-Validation Strategy**
  * A **5-fold Stratified Cross-Validation** approach was used.
  * In this method:
    * The dataset was divided into **five equal-sized folds**.
    * During each iteration, **four folds** were used for training and the **remaining fold** was used for testing.
    * This process was repeated **five times**, ensuring that every observation served as the test set exactly once.
    * The final performance was calculated by averaging the results from all five folds.
The **Stratified** approach preserves the original class distribution in every fold, making the evaluation more reliable for classification problems, especially when the classes are slightly imbalanced.

* **Evaluation Metric**
  * The models were evaluated using the **Receiver Operating Characteristic – Area Under the Curve (ROC-AUC)** score.
  * ROC-AUC measures how well a classifier distinguishes between positive and negative classes across all possible classification thresholds.
  * An AUC value closer to **1.0** indicates excellent classification performance, whereas an AUC close to **0.5** indicates performance similar to random guessing.

* **Models Compared**
  * The following classification models were evaluated using the same 5-fold Stratified Cross-Validation procedure:
    * **Logistic Regression**
    * **Controlled Decision Tree**
    * **Random Forest**
    * **Gradient Boosting**
Using the same evaluation strategy ensured a fair comparison between all machine learning algorithms.

* **Reported Results**
  * For each model, the following statistics were calculated:
    * **Mean ROC-AUC**
    * **Standard Deviation (Std) of ROC-AUC**
  * The **mean ROC-AUC** represents the average predictive performance across the five folds.
  * The **standard deviation** measures the consistency of the model's performance across different train-test splits.
    * **High Mean AUC** indicates strong overall predictive performance.
    * **Low Standard Deviation** indicates stable and consistent performance across different subsets of the data.
Together, these two statistics provide a more complete assessment of model quality.

* **Interpretation**
  * Cross-validation provides a more dependable estimate of a model's ability to generalize than a single train-test split.
  * A single split may produce overly optimistic or pessimistic results depending on how the data happens to be divided. Cross-validation reduces this uncertainty by evaluating the model multiple times on different subsets of the dataset.
  * As a result:
    * Performance estimates become more reliable.
    * Model comparison becomes fairer.
    * The influence of random data partitioning is significantly reduced.
    * The selected model is more likely to perform well on unseen real-world data.

* **Outcome**
  * The **5-fold Stratified Cross-Validation** experiment successfully compared the performance of **Logistic Regression**, **Controlled Decision Tree**, **Random Forest**, and **Gradient Boosting** using **ROC-AUC** as the evaluation metric. By reporting both the **mean** and **standard deviation** of ROC-AUC, the analysis provided a robust assessment of each model's predictive accuracy and stability. This comparison helped identify the model that offered the best balance between high classification performance and consistent generalization across multiple data splits.


### Task 6: Hyperparameter Tuning with GridSearchCV
To improve the performance of the Random Forest model, **hyperparameter tuning** was performed using **GridSearchCV**. Instead of manually selecting hyperparameter values, Grid Search systematically evaluated multiple combinations to identify the configuration that produced the best cross-validation performance.

* **Building a Machine Learning Pipeline**
  * A Scikit-learn **Pipeline** was created to combine all preprocessing and model training steps into a single workflow.
  * The pipeline consisted of the following stages:
    * 1. **Simple Imputer** – Replaces missing values in the dataset.
    * 2. **Standard Scaler** – Standardizes numerical features by removing the mean and scaling them to unit variance.
    * 3. **Random Forest Classifier** – Performs the final classification task.
Using a pipeline ensures that every preprocessing step is applied consistently during both training and cross-validation, preventing data leakage and making the workflow reproducible.

* **Defining the Hyperparameter Grid**
  * A grid of candidate hyperparameter values was created for the Random Forest model.
  * The following parameters were tested:
  * **Number of Trees (`n_estimators`)**
    * 50
    * 100
    * 200
  * **Maximum Tree Depth (`max_depth`)**
    * 5
    * 10
    * None (no depth limit)
  * **Minimum Samples per Leaf (`min_samples_leaf`)**
    * 1
    * 5
Each possible combination of these values was evaluated during the search process.

* **Performing Grid Search**
  * The **GridSearchCV** algorithm was used together with **5-fold Cross-Validation**.
  * For every hyperparameter combination:
    * The training data was divided into five folds.
    * The model was trained five times.
    * Each fold served as the validation set once.
    * The average cross-validation score was calculated.
The hyperparameter combination with the highest average performance was selected as the optimal model.

* **Total Number of Model Fits**
  * The total number of model evaluations performed by GridSearchCV was calculated as:
    * **3 values** for `n_estimators`
    * **3 values** for `max_depth`
    * **2 values** for `min_samples_leaf`
  * This results in:
    * **3 × 3 × 2 = 18 hyperparameter combinations**
  * Since **5-fold Cross-Validation** was used:
    * **18 × 5 = 90 total model fits**
Each of these 90 models was trained and evaluated automatically by GridSearchCV to identify the best-performing configuration.

* **Best Hyperparameters and Performance**
  * After the search process was completed, the following results were reported:
    * **Best Hyperparameter Combination**
    * **Best Cross-Validation Score**
  * These values represent the model configuration that achieved the highest average performance across all cross-validation folds.
  * The optimized model was then used for further evaluation and comparison with previously trained models.

* **Grid Search vs. Randomized Search** 
  * Grid Search performs an **exhaustive search**, meaning that every possible hyperparameter combination in the defined grid is evaluated.
  * **Advantages of Grid Search**
    * Guarantees evaluation of every specified combination.
    * Identifies the best configuration within the search space.
    * Suitable for small or moderate-sized parameter grids.
  * **Disadvantages of Grid Search**
    * Computationally expensive.
    * Training time increases rapidly as the number of parameters and candidate values grows.
    * May become impractical for very large search spaces.
  * **Randomized Search**, in contrast, evaluates only a randomly selected subset of hyperparameter combinations.
  * Advantages of Randomized Search include:
    * Much faster for large search spaces.
    * Requires fewer model evaluations.
    * Often finds near-optimal solutions with significantly lower computational cost.

* **Outcome**
  * The Random Forest model was successfully optimized using **GridSearchCV** within a complete preprocessing pipeline consisting of an **Imputer**, **StandardScaler**, and **Random Forest Classifier**. A total of **90 model fits** were evaluated through **5-fold Cross-Validation**, and the best hyperparameter combination along with its corresponding cross-validation score was identified. This experiment demonstrated the importance of systematic hyperparameter tuning and highlighted the trade-off between the exhaustive nature of **Grid Search** and the computational efficiency of **Randomized Search**.


### Task 7: Manual Learning Curve
To understand how the amount of training data affects model performance, a **manual learning curve analysis** was performed using the best machine learning pipeline obtained from the hyperparameter tuning stage. The model was trained on progressively larger portions of the training dataset, and its performance was evaluated at each stage.

* **Training on Different Dataset Sizes**
  * The optimized pipeline was trained using five different proportions of the training data:
    * **20%**
    * **40%**
    * **60%**
    * **80%**
    * **100%**
  * For each subset size, the model was trained from scratch and evaluated using both the training data and the independent test dataset.
  * This approach illustrates how model performance changes as more training examples become available.

* **Evaluation Metric** 
  * For every training subset, the following performance metric was calculated:
    * **Training ROC-AUC**
    * **Testing ROC-AUC**
  * The **Training ROC-AUC** measures how well the model fits the training data, while the **Testing ROC-AUC** measures how well the model generalizes to unseen data.
  * The results for all dataset sizes were summarized in a table (and optionally visualized using a learning curve plot).

* **Interpretation of the Learning Curve**
  * The learning curve provides valuable insight into the relationship between dataset size and model performance.
  * Training ROC-AUC
  * As the size of the training dataset increases:
    * The **Training ROC-AUC** typically decreases slightly.
    * This occurs because the model has more diverse data to learn from and is less able to memorize individual training examples.
    * A gradual decrease in training performance generally indicates **reduced overfitting**.

* **Testing ROC-AUC**
  * As additional training data becomes available:
    * The **Testing ROC-AUC** generally increases.
    * More training examples allow the model to learn broader and more representative patterns.
    * This leads to improved **generalization** and more reliable predictions on unseen data.

* **Determining Whether the Model is Data-Limited or Capacity-Limited**
  * The shape of the learning curve helps diagnose whether further improvements are likely to come from collecting more data or from using a more powerful model.
  * **Data-Limited Model**
    * If the **Testing ROC-AUC continues to increase at 100% of the available training data**, it suggests that the model has not yet reached its full potential.
  * In this case:
    * Additional training data would likely improve performance.
    * The model is considered **data-limited**.
  * **Capacity-Limited Model**
    * If the **Testing ROC-AUC plateaus** and no longer improves as more data is added, it indicates that the current model has reached its learning capacity.
  * In this situation:
    * Collecting more data may provide little benefit.
    * Performance improvements are more likely to come from using a more sophisticated model, better feature engineering, or additional hyperparameter tuning.
    * The model is considered **capacity-limited**.

* **Importance of Learning Curves**
  * Learning curves are valuable diagnostic tools because they help identify:
    * Whether the model is overfitting or underfitting.
    * Whether additional training data is likely to improve performance.
    * Whether increasing model complexity is necessary.
    * Whether the current model is making efficient use of the available data.
These insights support informed decisions about future model development and optimization.

* **Outcome**
  * The manual learning curve experiment demonstrated how model performance changes as the amount of training data increases. By comparing **Training ROC-AUC** and **Testing ROC-AUC** across training subsets ranging from **20% to 100%**, the analysis provided insight into the model's learning behavior. The learning curve also helped determine whether the model was **data-limited**, suggesting that more data could improve performance, or **capacity-limited**, indicating that improvements would require a more expressive model or enhanced feature engineering rather than additional training data.


### Task 8: Serialize Best Model
After identifying the best-performing machine learning pipeline through hyperparameter tuning and evaluation, the final model was **serialized** and saved to disk. Model serialization allows the trained pipeline to be reused later without repeating the computationally expensive training process.

* **Saving the Best Model**
  * The optimized machine learning pipeline was saved using the **Joblib** library.
  * The following function was used:
    * joblib.dump(best_pipeline, 'best_model.pkl')
  * This command stores the complete trained pipeline—including preprocessing steps (such as imputation and scaling) and the trained Random Forest model—into a file named       **`best_model.pkl`**.
  * Saving the entire pipeline ensures that future predictions follow exactly the same preprocessing workflow used during training.

* **Reloading the Saved Model**
  * To verify that the saved model was correctly stored, it was reloaded from disk using the following command:
    * joblib.load('best_model.pkl')
  * The reloaded pipeline was then used to generate predictions on a small set of sample observations.
  * The predictions produced by the loaded model matched those of the original trained pipeline, confirming that the serialization and deserialization process was successful.

* **Verification**
  * After loading the serialized model:
    * Sample rows from the dataset were passed to the pipeline.
    * Predictions were generated successfully.
    * The prediction results were compared with those obtained before saving the model.
    * The outputs were consistent, demonstrating that the model had been restored correctly.
This verification step confirmed that the saved model could be reliably reused.

* **Importance of Model Serialization**
  * Model serialization provides several practical advantages:
    * Eliminates the need to retrain the model every time predictions are required.
    * Preserves the complete trained pipeline, including preprocessing and model parameters.
    * Ensures reproducibility by allowing identical predictions across different sessions.
    * Enables easy deployment in production environments such as web applications, APIs, or cloud services.
    * Reduces computational cost and saves time during inference.
Serialization is an essential step in the machine learning deployment workflow because it transforms a trained model into a reusable production asset.

* **Outcome**
  * The best-performing machine learning pipeline was successfully serialized using **`joblib.dump()`** and stored as **`best_model.pkl`**. The model was then reloaded using **`joblib.load()`**, and its predictions were verified on sample data to confirm correctness. This process demonstrated that the trained pipeline can be reused without retraining, ensuring reproducibility, efficient deployment, and consistent prediction performance in future applications.
