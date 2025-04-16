# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything from current folder
COPY . .

# Install Python packages
RUN pip install --no-cache-dir flask pandas scikit-learn matplotlib seaborn

# Expose Flask default port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
