FROM python:3.11-slim

# Copy the pyproject.toml and install hatchling
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install hatchling && hatch build && pip install dist/*.whl

# Set the command to launch Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]