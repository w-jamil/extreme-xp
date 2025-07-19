# Batch, Continual & Onlin Learning Evaluation Suite

This project contains three distinct experiments for evaluating online machine learning algorithms: Task-CL, Task-Domain-CL, and a pure Online evaluation. All experiments can be run automatically with a single command using Docker.

## Prerequisites

-   **Docker Desktop**: You must have Docker installed and running on your system. You can download it from the [official Docker website](https://www.docker.com/products/docker-desktop/).

## Setup Instructions

**Step 1: Download and Unzip**

Download the project as a single ZIP file and unzip it into a folder on your computer.

**Step 2: Place Your Data Files**

This project contains three experiment folders. You must place the correct `.parquet` data files inside the `cyber/` sub-directory of each one:

-   `task_cl/cyber/`
-   `task_domain_cl/cyber/`
-   `online/cyber/`

Place the relevant datasets for each experiment into its corresponding folder.

## How to Run All Experiments

**Step 1: Open a Terminal**

Open your command line tool (Terminal on macOS/Linux, PowerShell or Command Prompt on Windows).

**Step 2: Navigate to the Project Folder**

Use the `cd` command to navigate into the main project folder you unzipped (the one containing the `docker-compose.yml` file).

```bash
cd path/to/your/project-folder
```

**Step 3: Run the Single Command**

Execute the following command. This will automatically build the environment and run all three experiments in sequence. The first time you run this, it may take several minutes to download the Python image and install libraries.

```bash
docker-compose up --build
```

You will see output from the `tacl_sim.py`, `task_domain_sim.py`, and `online.py` scripts in your terminal. Wait for the command to finish completely.

## Finding the Results

Once the command has finished, the output CSV files will be located inside the `results/` folder of each corresponding experiment directory.

-   Look in **`task_cl/results/`** for `tacl_results.csv`.
-   Look in **`task_domain_cl/results/`** for `task_domain_results.csv`.
-   Look in **`online/results/`** for `online_evaluation_results.csv`.

You can open these files with any spreadsheet software to analyze the results.
