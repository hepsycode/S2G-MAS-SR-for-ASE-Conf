# 00 - S2G-MAS-SR: Multi-Agent Framework

This folder contains the implementation of the **S2G-MAS-SR** framework for automated search string generation in SLRs/SMSs.

## Components

- `00 - S2G-MAS-SR-NoRag.ipynb`: Jupyter Notebook to run the framework **without RAG**.
- `00 - S2G-MAS-SR-Rag.ipynb`: Jupyter Notebook to run the framework **with RAG**.
- `config/`: Configuration files (model parameters, API keys, LangGraph settings).
- `requirements.txt`: Required Python packages.
- `00-Query`: Logs and generated outputs.

## Architecture Overview

The MAS orchestrates a network of agents via LangGraph:
- **Query Generation Agent**
- **Router Agent**
- **Self-Reflection RAG Agent**
- **Search String Formatter Agent**
- **Database Runner Agent**
- **Result Evaluation Agent**

## How to Run

- ### Install dependencies:

Dependencies are listed in the `requirements.txt` file.

- ### Configure your API keys

Edit your API keys and parameters in:

```
config/
```

- ### Run the framework in **Jupyter Notebooks (recommended)**

Launch Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```

Then open and run one of the following:

- **Without RAG**: 
  - `00 - S2G-MAS-SR-NoRag.ipynb`

- **With RAG**:  
  - `00 - S2G-MAS-SR-Rag.ipynb`

- ### Outputs

Logs and generated outputs will be stored in:

```
/00 - Query/
```

---

## Sustainability Tracking

The framework integrates **CodeCarbon** for monitoring:
- Energy consumption (CPU/GPU/RAM)
- CO2 emissions
- Execution time

Results are logged in the output folder.

---
