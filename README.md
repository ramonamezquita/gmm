# Gaussian Mixture Model CLI

This CLI program fits a Gaussian Mixture Model (GMM) to data, evaluates different numbers of components, and selects the best model based on the Akaike Information Criterion (AIC).

## Features
- Fits a Gaussian Mixture Model (GMM) to data from a CSV file.
- Automatically determines the best number of components (`K`) within a specified range.
- Outputs the best GMM parameters, AIC score, and log-likelihood.

---

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:ramonamezquita/gmm.git
   cd gmm
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the CLI with the following command:

```bash
python analyze.py --filename <path_to_csv> [options]
```

### Required Arguments
- `--filename`, `-F`: Path to the CSV file containing the dataset.  
  The file should contain numeric data as comma-separated values.

### Optional Arguments
- `--start`, `-S`: The starting value of `K` (number of components). Default is `2`.
- `--end`, `-E`: The ending value of `K`. Default is `10`.

---

### Example

Suppose you have a CSV file `data.csv` containing the following data:

```text
1.0,2.0,3.0
4.0,5.0,6.0
```

You can run the program like this:

```bash
python analyze.py --filename data.csv --start 2 --end 5
```

The program will output the best model based on the AIC score, including the log-likelihood and other parameters.

---
