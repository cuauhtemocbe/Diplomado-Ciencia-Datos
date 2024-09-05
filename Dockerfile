FROM python:3.12.3-slim
RUN mkdir workspace
WORKDIR /workspace
COPY . .
RUN pip install "poetry"
RUN poetry config virtualenvs.create false
ENV PYTHONPATH=/workspace/src
RUN poetry install --no-root --without dev
RUN rm -R *

# EXPOSE 5000
# CMD ["gunicorn", "src.app_clustering.app:app", "--bind", "0.0.0.0:5000"]
CMD ["/bin/bash"]