# mmWave Radar Point Cloud Processing Pipeline

This project is a Python-based processing pipeline for mmWave radar data. It takes raw ADC data (`.bin` files), processes it to detect objects, and generates a 3D point cloud visualization video (`.mp4`).

## How to Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```

2.  **Download the Data**
    * The download link for the required raw `.bin` data file is located in the text file:
        `bin files/download_link.txt`
    * Download the data and place the `.bin` file(s) inside the `bin files/` folder.

3.  **Install Requirements**
    * It is recommended to use a virtual environment.
    * Install all necessary Python libraries from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Pipeline**
    * The `main.py` script will run the entire process from start to finish.
    ```bash
    python main.py
    ```

5.  **Get Output**
    * **Processed Data:** The intermediate point cloud data will be saved as `.csv` files in the `csv/` folder.
    * **Final Video:** The final 3D animations will be saved as `.mp4` files in the `vid/` folder.

## Project Structure & Key Files

* `main.py`: The main executable script that runs the entire pipeline. It calls `main_processing.py` first, then `maker_3D.py`.

* `main_processing.py`: (Formerly `Raw_Csv.py`) This is the **core processing logic** of the project. It reads the raw `.bin` data, performs the Range-FFT, Doppler-FFT, CFAR detection, and Angle-FFT to detect objects and generate a point cloud.

* `maker_3D.py`: This script reads the processed `.csv` files and uses `matplotlib` to generate and save the final 3D point cloud animations.

* `resources/`: This folder contains the original C code. The logic in these files was used as a reference and guide to create the Python-based processing algorithms in `main_processing.py`.

* `bin files/`: This folder is for the raw `.bin` data files. It contains a `download_link.txt` with a link to the sample data.

* `csv/`: This is the output folder for the processed `.csv` point cloud data.

* `vid/`: This is the final output folder for the generated `.mp4` videos.
