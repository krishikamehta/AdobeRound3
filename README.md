# Document Analyzer Web Application

This is a persona-driven document intelligence system built as a web application. It takes a folder of PDF documents as input, along with a user persona and a specific task. The backend analyzes the documents, extracts relevant sections based on a hybrid ranking model (lexical and semantic), and provides a refined, human-readable summary.

The application is packaged in a Docker container for easy setup and portability.

***

### Key Features

* **PDF Analysis**: Processes a collection of PDF files to understand their structure and content.

* **Persona-Driven Ranking**: Ranks document sections based on a user-defined persona and task, ensuring the most relevant information is prioritized.

* **Hybrid Scoring**: Uses a combination of keyword matching (BM25) and semantic similarity (Sentence Transformers) for accurate relevance scoring.

* **Extractive Summarization**: Generates a refined, concise summary of the most important sections to provide quick insights.

* **Containerized Deployment**: Packaged in a Docker container, making it easy to build, share, and run on any system.

***

### Technologies Used

* **Frontend**: HTML, CSS, JavaScript (for the user interface and API calls)

* **Backend**: Python, Flask, Gunicorn (for the web server and API logic)

* **Document Processing**: PyMuPDF, NLTK, Sentence Transformers, Rank BM25

* **Deployment**: Docker, NGINX (as a reverse proxy and static file server)

***

### Getting Started

These instructions will help you get a copy of the project up and running on your local machine.

#### Prerequisites

* **Docker**: Ensure Docker is installed and running on your system.

#### Installation and Setup

1.  **Clone the Repository** (or create the project directory as guided).

2.  **Navigate to the Project Directory**:

    ```
    cd /path/to/your/doc_analyzer_web
    ```

3.  **Build the Docker Image**: This command will read the `Dockerfile`, install all dependencies, and create a single, self-contained image named `document-analyzer`.

    ```
    docker build -t document-analyzer .
    ```

4.  **Run the Docker Container**: This command starts the application and maps port 80 on your machine to port 80 in the container, making the application accessible via your web browser.

    ```
    docker run -p 80:80 document-analyzer
    ```

    You will see logs from both Gunicorn and NGINX in your terminal.

***

### Usage

1.  Open your web browser and go to `http://localhost`.

2.  You will be presented with a form.

3.  **Upload the Folder**: Click the "Upload Folder" button and select the directory containing your PDF documents.

4.  **Enter Persona and Task**: Fill in the `Persona` and `Task` fields to guide the analysis.

5.  **Analyze**: Click "Analyze Documents". The page will display the top-ranked sections and a detailed sub-section analysis in a human-readable format.

***

### Troubleshooting

* **`ModuleNotFoundError`**: This error occurs when a required Python package is missing. Ensure the `Dockerfile`'s `requirements.txt` generation line includes all necessary packages: `flask`, `flask-cors`, `gunicorn`, `nltk`, `sentence-transformers`, `requests`, `PyMuPDF`, and `rank-bm25`.

* **`Read timed out`**: This indicates a poor or slow internet connection during the build process. The download of the large language model can take a while. Ensure your network is stable before rebuilding.

* **Build hangs or is stuck**: The build might be waiting for a large file to download. Be patient. If it's a persistent issue, a network problem is likely. Try restarting your network connection.

* **`ModuleNotFoundError: No module named '...'` in the `docker run` log**: This means the Python package was not installed correctly or is missing from the `requirements.txt` list. The build itself might have completed successfully, but the container failed to run. Double-check your `Dockerfile` to ensure all packages are listed.
