# TuanLe_Bayes-DSS-A-Clinical-Decision-Support-Module-for-Comparing-Treatment-Policies-
Bayes-DSS is a Clinical Decision Support module that uses Bayesian Decision Theory to compare breast cancer treatment policies. It integrates AI diagnostic priors and historical error rates with a 2-compartment tumor simulation to calculate Expected Utility, providing a mathematically robust, risk-averse safety net for clinical decision-making.

## I. Introduction
**Bayes-DSS** is a Clinical Decision Support System (CDSS) module that uses Bayesian Decision Theory to evaluate and compare cancer treatment policies. Instead of blindly trusting an AI classifier's top prediction, this module translates the AI's probabilistic uncertainty directly into human-understandable clinical consequences (tumor reduction vs. drug toxicity). 

**Flexibility & Constraints:**
This module is agnostic to the deep learning architecture; it can work with **any AI classification model** that produces predictive priors (softmax probabilities). In this repository, it natively ingests outputs from a custom multimodal Xception diagnostic model. 

However, please note that the biological parameters pre-configured in this model are **specifically designed for a 5-class breast cancer problem** representing the following subtypes:
1. Benign
2. Luminal A
3. Luminal B
4. HER2-enriched (HER2+)
5. Triple-Negative (TN)

If you wish to utilize this module for different cancer types or subtypes, you must manually update the biological constants (growth rates and drug efficacy) in the configuration dictionary to reflect the appropriate medical literature for those specific tumors.

---

## II. Model Architecture
The architecture of Bayes-DSS decouples the heavy image-processing deep learning pipeline from the decision analysis, ensuring rapid simulation and sensitivity testing. As illustrated in the system diagram, the workflow consists of 6 sequential blocks:

1. **Data Ingestion:** Loads the AI's raw diagnostic probabilities (`y_pred`) and the actual ground truth data (`y_test`) to generate the historical Confusion Matrix, which empirically quantifies the AI's error rate.
2. **Simulation Configuration:** An interactive dashboard to set critical variables without hardcoding. This includes Days per cycle ($D$), Max cycles, a range of toxicity penalty weights ($\lambda$), and the specific biological constants for each tumor subtype.
3. **Tumor Dynamic Simulation (The Physics Engine):** Sets up the treatment policies (Aggressive, Adaptive, Intermittent) and performs mathematical unit conversion (cycle to daily rates). It simulates daily biological tumor growth ($S_{next}, R_{next}$) and tracks toxicity accumulation. It strictly enforces a "clinical lock," meaning the virtual doctor evaluates the tumor and locks in the drug dose ($a_t$) only on Day 1 of a clinical cycle (e.g., every 21 days), perfectly mirroring real-world oncology practices.
4. **Utility Matrix Setup:** Calculates the deterministic clinical value $V$ for each treatment policy against each possible tumor subtype. The formula calculates the trade-off between the final tumor burden and the accumulated toxicity: $V = -(\text{burden}) - (\lambda \times \text{sum} \_ \text{at})$.
5. **Patient Bayesian Evaluation (The Decision Logic):** Retrieves the patient's individual diagnostic probabilities and updates them using the conditional probabilities from the Confusion Matrix. It then calculates the Expected Utility ($EU$) via the dot product of these joint probabilities and the Utility Matrix, mathematically hedging against AI misclassification.
6. **Decision Extraction:** Extracts the multi-dimensional $EU$ grid for each patient. It scans the grid to find the specific cycle length yielding the maximum Expected Utility for each policy option, ultimately flagging the mathematically absolute best strategies (including ties) to give doctors comparable alternatives.

![Module 2 Architecture](module-2-architecture.png)

---

## III. Mathematical Foundations: Tumor Dynamics and Decision Theory
The core logic of Bayes-DSS relies on two mathematical frameworks: a 2-compartment differential model for tumor biology, and Bayesian utility for risk mitigation.

### 1. Discrete 2-Compartment Tumor Model
Based on the Lotka-Volterra competition framework and adapted for discrete-time clinical steps, the tumor is divided into Sensitive ($S$) and Resistant ($R$) cells. Assuming the tumor is small enough relative to its carrying capacity ($K$) to omit the logistic saturation factor temporarily, the daily biological update equations are:

$$S_{t+1} = S_t \cdot (1 + r_s) \cdot (1 - e_s \cdot a_t)$$
$$R_{t+1} = R_t \cdot (1 + r_r) \cdot (1 - e_r \cdot a_t)$$

*   **$S_t, R_t$**: Volume of sensitive and resistant cells.
*   **$r_s, r_r$**: Intrinsic daily growth rates.
*   **$e_s, e_r$**: Daily drug efficacy (kill fraction). 
*   **$a_t \in [1]$**: The clinical dose intensity administered at time $t$.

### 2. Clinical Value Function
The physical outcome of a policy $\pi$ on a specific tumor subtype $s$ after $T$ total cycles is evaluated using a Value function $V$. It calculates the trade-off between the final tumor burden and the cumulative toxicity suffered by the patient:

$$V(\pi_i | s_j) = -(S_T + R_T) - \lambda \sum_{t=0}^{T} a_t$$

### 3. Bayesian Expected Utility ($EU$)
To act as a "safety net" against AI misclassification, the system does not calculate simple expected value. It updates the AI's prior probability $p(s)$ using the historical conditional probability $P(t | s)$ (the chance the tumor is truly $t$ given the AI predicted $s$, derived directly from the confusion matrix).

$$EU(\pi | s) = \sum_{t} P(t | s) V(\pi | t)$$

This guarantees that if the AI is uncertain, the model mathematically hedges its bets toward a safer, less toxic treatment policy.

---

## IV. Constants and Parameters
To ensure clinical credibility, the variables in this model are not arbitrary; they are mapped directly from published oncological and pharmacoeconomic literature.

*   **Biological Constants ($S_0, R_0, r_s, r_r, e_s, e_r$):** 
    *   Intrinsic growth rates ($r_s$) are derived from clinical Tumor Volume Doubling Time (TVDT) studies (e.g., Zhang et al., 2017). Resistant cells are modeled to grow at 50% the rate of sensitive cells due to fitness costs.
    *   Drug efficacy rates ($e_s, e_r$) are derived from pathologic complete response (pCR) tracking in neoadjuvant chemotherapy studies (e.g., Ubezio & Cameron, 2008). 

| Tumor Subtype | $S_0$ (Initial Sensitive) | $R_0$ (Initial Resistant) | $r_s$ (Daily Growth) | $r_r$ (Daily Growth) | $e_s$ (Efficacy / Cycle) | $e_r$ (Efficacy / Cycle) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Benign** | 1.00 | 0.00 | 0.00076 | 0.00000 | 0.00 | 0.00 |
| **Luminal A** | 0.95 | 0.05 | 0.00270 | 0.00135 | 0.20 | 0.0325 |
| **Luminal B** | 0.90 | 0.10 | 0.00330 | 0.00165 | 0.35 | 0.0055 |
| **HER2+** | 0.825 | 0.175 | 0.00380 | 0.00304 | 0.60 | 0.125 |
| **Triple-Negative** | 0.60 | 0.40 | 0.00550 | 0.00440 | 0.55 | 0.085 |

*(Note: The simulation engine automatically converts the per-cycle drug efficacy constants into daily kill fractions based on the user's selected cycle length $D$)*

*   **Treatment Cycle Length ($D$):** Set to 21 days by default, mapping to standard neoadjuvant regimens evaluated in clinical practice (Keam et al., 2017).
*   **Toxicity Penalty Weight ($\lambda$):** Represents the "Disutility of Treatment". Instead of using a single fixed number, the model sweeps across a range of $\lambda$ values (e.g., 0.01 to 1.0) to perform **Decision Curve Analysis (DCA)**. A low $\lambda$ (e.g., 0.05) mimics mild side effects or a highly resilient patient, while a high $\lambda$ (>0.15) mimics severe Grade 3 toxicity (Peasgood et al., 2010; Lloyd et al., 2006).

---

## V. Result & Interpretations
The output of the module is an extensive multi-dimensional grid exported as a CSV file (e.g., `optimal_treatments_full_grid.csv`). 

![Bayes DSS Sample](bayes-dss-sample.png)

### How to Read the Output Data
The CSV contains the following columns: `Patient_ID`, `True_Subtype`, `Predicted_Subtype`, `Lambda_Weight`, `Policy_Option`, `Recommended_Cycles`, `Expected_Utility`, `Is_Optimal_Choice`.

*   **Sensitivity Analysis via Lambda:** As you look down the rows for a single patient, you will see the `Lambda_Weight` increase. At low lambda (e.g., 0.01), the model acts aggressively, prioritizing tumor reduction and recommending extended cycles. As lambda crosses critical thresholds, the mathematical penalty of toxicity outweighs the oncological benefit, and the model correctly recommends halting treatment early (1 cycle).
*   **Understanding Ties (⭐ YES vs Alternative):** The model explicitly flags all mathematically optimal choices. If both "Aggressive" and "Adaptive" policies are flagged as `⭐ YES` for 1 cycle, it means the tumor biology has not yet crossed the threshold required for the Adaptive policy to reduce the dose. Their biological outcomes and toxicity penalties are identical up to that point.
*   **Diagnostic Mismatches:** By filtering for patients where `True_Subtype != Predicted_Subtype`, you can observe the Bayesian Safety Net in action. You will see how the Expected Utility defaults to safer, compromised regimens when the AI makes a highly uncertain prediction.

---

## Installation and Usage Guide

### Prerequisites
```bash
pip install numpy pandas scikit-learn ipywidgets
Inputs Required
The module is designed to run in a Jupyter/Colab environment using ipywidgets for interactive configuration. It requires two primary data inputs, typically stored in a .npz file from your AI testing script:
y_pred: An array of shape (num_patients, num_classes) containing the softmax probability priors for each patient.
y_test: An array containing the ground-truth labels, used exclusively to generate the confusion_matrix (which simulates the AI's historical track record).
Note: In this specific repository, the model ingests output arrays from a custom multimodal Xception network. Users wishing to apply this DSS to their own classifiers only need to replace the y_pred and y_test variables in the Data Ingestion block with their own numpy arrays.
Usage
Run the script in a Jupyter Notebook.
Adjust the clinical boundaries in the generated UI (Days per cycle, maximum cycles, lambda range steps).
Click "Run Simulation & Extract All".
The system will compute the continuous biological physics, apply the Bayesian transformations, and output a CSV dataframe containing the comparative Expected Utility of every policy for every patient.

--------------------------------------------------------------------------------
References
The mathematical modeling and parameter calibrations used in this framework are supported by the following literature:
Ubezio, P., & Cameron, D. (2008). Cell killing and resistance in pre-operative breast cancer chemotherapy. BMC Cancer, 8(1), 201.
Zhang, S., et al. (2017). Correlation Factors Analysis of Breast Cancer Tumor Volume Doubling Time Measured by 3D-Ultrasound. Medical Science Monitor, 23, 3147–3153.
Peasgood, T., et al. (2010). Impact of breast cancer on health-related quality of life.
Lloyd, A., et al. (2006). Health state utility values for breast cancer.
Gatenby, R. A., et al. (2009). Adaptive therapy. Cancer research, 69(11).
Keam, B., et al. (2017). (Source for standardizing the 21-day clinical chemotherapy cycle D).
