all: 
	docker compose up --build -d

stop: 
	docker compose down

evaluation:
	docker compose up -d --build evaluation

traffic:
	docker run -it --network promgraf-mlops-exam-student_monitoring -v ./data:/data promgraf-mlops-exam-student-evaluation:latest python3 traffic.py

fire-alert:
	echo "a voir"
	
