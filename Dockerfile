# Start from a Python base image, which includes a full Python environment
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install NGINX and necessary tools using apt-get
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install all Python dependencies, including gunicorn
#RUN echo "flask\nflask-cors\ngunicorn\nnltk\nsentence-transformers\nrequests" > requirements.txt && \
    #for i in $(seq 1 3); do \
      #pip install --no-cache-dir --default-timeout=120 -r requirements.txt && break; \
     # sleep 5; \
    #done
RUN echo "flask\nflask-cors\ngunicorn\nnltk\nsentence-transformers\nrequests\nPyMuPDF\nrank-bm25" > requirements.txt && \
    for i in $(seq 1 3); do \
      pip install --no-cache-dir --default-timeout=120 -r requirements.txt && break; \
      sleep 5; \
    done

# Pre-download NLTK data and the sentence-transformers model
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')" && \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('models/all-MiniLM-L6-v2')"

# Copy the backend code and the NGINX configuration
COPY api.py core_logic.py nginx.conf ./

# Copy the frontend files
COPY index.html /var/www/html/

# Update the NGINX configuration to reflect the file paths
RUN sed -i 's|/app/frontend|/var/www/html|g' nginx.conf && \
    sed -i 's|localhost:8000|127.0.0.1:8000|g' nginx.conf && \
    cp nginx.conf /etc/nginx/nginx.conf

# Expose port 80 (where NGINX listens)
EXPOSE 80

# Command to start both NGINX and Gunicorn
CMD sh -c "gunicorn --bind 127.0.0.1:8000 api:app & nginx -g 'daemon off;'"
