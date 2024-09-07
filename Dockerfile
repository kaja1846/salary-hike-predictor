# 1. Use a base image with Python 3.9
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code into the container
COPY . /app

# 6. Define the command to run the Python script
CMD ["python", "app.py"]