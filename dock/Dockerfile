# Dockerfile

# 1. Use an offical Python runtime as a base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (for caching efficiency)
COPY requirements.txt .

# 4. Install depedencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code into the container
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run the application
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
