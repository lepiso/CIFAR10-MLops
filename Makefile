.PHONY: help install train test api streamlit docker-build docker-up clean

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Train the model"
	@echo "  make test        - Run tests"
	@echo "  make api         - Run FastAPI server"
	@echo "  make streamlit   - Run Streamlit app"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up   - Start Docker containers"
	@echo "  make clean       - Clean temporary files"

install:
	pip install -r requirements.txt
	pip install -r requirements-streamlit.txt

train:
	python src/train_simple.py

test:
	pytest tests/ -v --cov=app --cov-report=html

api:
	uvicorn app.main:app --reload --port 8000

streamlit:
	streamlit run streamlit_app/app.py

docker-build:
	docker build -t cifar10-api -f Dockerfile .
	docker build -t cifar10-streamlit -f Dockerfile.streamlit .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/ reports/figures/ reports/metrics/
