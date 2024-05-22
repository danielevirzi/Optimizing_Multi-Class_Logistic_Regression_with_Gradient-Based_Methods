# Optimizing Multi-Class Logistic Regression with Gradient-Based Methods
## HOMEWORK OPTIMIZATION FOR DATA SCIENCE 2024  
**MULTI-CLASS LOGISTIC REGRESSION**  

Consider a Multi-Class Logistic problem of the form: 

$$\tag{1} \min_{X \in \mathbb{R}^{d \times k}} \sum_{i=1}^{m} \left[ -x_{b_i}^T a_i + log\left( \sum_{c=1}^{k} \exp(x_{c}^T a_i) \right) \right] $$

Likelihood for single training example $i$ with features $a_i \in \mathbb{R}^{d}$ and label $b_i \in \{1, 2, \ldots, k\}$ is given by  

$$ \tag{2} P(b_i | a_i, X) = \frac{\exp(x_{b_i}^T a_i)}{\sum_{c=1}^k\exp(x_c^T a_i)}$$

where $x_c$ is column $c$ of matrix parameter $X \in \mathbb{R}^{d \times k}$ to maximize likelihood over $m$ i.i.d. training samples.

**HOMEWORK**
1. Randomly generate a $1000 \times 1000$ matrix with entries from a $\mathcal{N}(0,1)$.
2. Generate $b_i \in \{1, 2, \ldots, k\}$ with $k = 50$ by computing $AX + E$ with $X \in \mathbb{R}^{d \times k}$, $E \in \mathbb{R}^{m \times k}$ sampled from Normal distribution and consider max index row as class label.
3. Solve problem $(1)$ with:
4.           - *Gradient Descent*.
5.           - *BCDG with Randomized rule*.
6.           - *BCDG with Gauss-Southwell rule*.
7. Choose a pubicly available dataset and test methods on this.
8. Analyze *Accuracy vs CPU Time*.
9. Describe what you did on a PDF file.
10. Submit project.
