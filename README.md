# Optimizing Multi-Class Logistic Regression with Gradient-Based Methods

## Project Structure

```
|-- data/                          # Contains dataset information and data files
|-- code/                          # Directory for all code implementations
|   |-- cpu/                          # CPU-based implementation
|   |   |-- experiment.ipynb             # Jupyter Notebook for CPU execution
|   |-- gpu/                          # GPU-based implementation
|   |   |-- experiment.py                # Python script for GPU execution
|   |-- test/                         # Validation and testing scripts
|   |   |-- test.ipynb                   # Notebook comparing GPU and CPU implementations
|-- report.pdf                     # Detailed report of results
```

## Task

### Multi-Class Logistic Regression

Consider a Multi-Class Logistic problem of the form:

$$ \tag{1} \min_{X \in \mathbb{R}^{d \times k}} \sum_{i=1}^{m} \left[ -x_{b_i}^T a_i + \log\left( \sum_{c=1}^{k} \exp(x_{c}^T a_i) \right) \right] $$

The likelihood for a single training example $i$ with features $a_i \in \mathbb{R}^{d}$ and label $b_i \in \{1, 2, \ldots, k\}$ is given by:

$$ \tag{2} P(b_i | a_i, X) = \frac{\exp(x_{b_i}^T a_i)}{\sum_{c=1}^k\exp(x_c^T a_i)} $$

where $x_c$ is column $c$ of the matrix parameter $X \in \mathbb{R}^{d \times k}$, and the objective is to maximize likelihood over $m$ i.i.d. training samples.


1. Randomly generate a $1000 \times 1000$ matrix with entries from a $\mathcal{N}(0,1)$.
2. Generate $b_i \in \{1, 2, \ldots, k\}$ with $k = 50$ by computing $AX + E$ where:
   - $X \in \mathbb{R}^{d \times k}$
   - $E \in \mathbb{R}^{m \times k}$ sampled from a Normal distribution
   - The maximum index in each row is taken as the class label.
3. Solve problem $(1)$ using:
   - *Gradient Descent*
   - *Block Coordinate Descent with a Randomized rule (BCDG-Random)*
   - *Block Coordinate Descent with the Gauss-Southwell rule (BCDG-GS)*
4. Choose a publicly available dataset and test the methods on it.
5. Analyze *Accuracy vs CPU Time*.

## Results

For detailed results and analysis on CPU Implementation, please take a look at the [report.pdf](./report.pdf).
The best result we achieved is 92.3% accuracy with a training of 0.16 s with an RTX 3070.


