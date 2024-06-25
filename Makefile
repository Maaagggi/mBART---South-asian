.PHONY: start-backend start-frontend kill-flask kill-react

start-backend: kill-flask
	@echo "Starting Flask backend..."
	FLASK_ENV=development flask run --host=0.0.0.0 --port=8000 &

start-frontend: kill-react
	@echo "Starting React frontend..."
	cd frontend && npm start

start: start-backend start-frontend
	@echo "Application started."

kill-flask:
	@echo "Attempting to kill any process using port 8000..."
	-fuser -k 8000/tcp  # Attempt to kill any process using port 8000
	@echo "Port 8000 is now free."

kill-react:
	@echo "Attempting to kill any process using port 3000..."
	-fuser -k 3000/tcp  # Attempt to kill any process using port 3000
	@echo "Port 3000 is now free."
