# Batch, Online and Continual Learning Evaluation

This project contains a suite of four distinct experiments for evaluating batch, online and continual machine learning algorithms. All code, dependencies, and data are automatically managed by Docker, allowing you to run everything with a single command.

On the first run, the necessary datasets will be automatically downloaded from a Zenodo repository and prepared for the experiments.

## 1. System Requirements

Before you begin, you **must** have **Docker Desktop** installed and running on your system.

-   **Operating System:** Windows 10/11 or modern macOS/Linux.
-   **Software:** **Docker Desktop**. If you do not have it, download it from the official website:
    -   [**Download Docker Desktop**](https://www.docker.com/products/docker-desktop/)

> **IMPORTANT:** After installing Docker Desktop, you must **open the application and wait for it to start completely**. Look for the small whale icon in your system tray (Windows) or menu bar (macOS). The icon must be stable (not animating) to indicate that Docker is running.

## 2. How to Run All Experiments

Follow these steps exactly to run the full suite.

### Step 1: Download and Unzip the Project

Download the project as a single ZIP file and unzip it into a location on your computer. This will create a main project folder containing all the necessary files. **You do not need to download or place any data files manually.**

### Step 2: Open a Terminal

Open your command line tool (e.g., Terminal on macOS, or PowerShell/Command Prompt on Windows).

> **Tip:** It's best to open a **new** terminal *after* you have installed and started Docker Desktop.

### Step 3: Navigate to the Project Folder

Use the `cd` command to navigate into the main project folder you just unzipped. This is the folder that contains the `docker-compose.yml` file.

```bash
# Example on Windows:
cd C:\Users\YourUser\Downloads\master-experiment-suite

# Example on macOS/Linux:
cd /Users/youruser/Downloads/master-experiment-suite
```

### Step 4: Run the Single Command

Execute the following command in your terminal. This will automatically build the environment, download data (if needed), and run all four experiments in sequence.

```bash
docker-compose up --build
```

> **What to Expect:** The first time you run this command, it will be slow as it needs to download the Python environment and the large dataset archives. You will see a lot of text scrolling in your terminal as each experiment runs. This is normal. Subsequent runs will be much faster as all data and dependencies will be stored locally.

## 3. Finding the Results

Once the command has finished, the output CSV files containing the results will be located inside the `results/` folder of each corresponding experiment directory.

-   Look in **`task_cl/results/`** for `tacl_results.csv`.
-   Look in **`task_domain_cl/results/`** for `task_domain_results.csv`.
-   Look in **`online/results/`** for `online_evaluation_results.csv`.
-   Look in **`batch_learning/results/`** for `batch_learning_results.csv`.

You can now open these files with Excel, Google Sheets, or any other spreadsheet software to analyze the results.

## 4. Troubleshooting Common Issues

-   **Error: `docker: The term 'docker' is not recognized...`**
    This means Docker is not installed or the Docker Desktop application is not running.
    1.  **Check:** Is the Docker Desktop application open and the whale icon stable?
    2.  **Restart Terminal:** Close your current terminal and open a brand new one. This usually fixes the issue after a fresh installation.

-   **Error: `'docker-compose' is not recognized...` or `'compose' is not a docker command`**
    This is a common issue with different Docker versions.
    **Try the other command syntax.** If `docker-compose up --build` failed, try `docker compose up --build` (without the hyphen), or vice-versa. One of them will work.

-   **Download Fails**
    Ensure you have a stable internet connection. If the Zenodo link is down, you may need to check the `ZENODO_ARCHIVE_URL` variable in the relevant Python script (`tacl_sim.py`, etc.).
