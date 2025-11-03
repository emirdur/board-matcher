#!/bin/bash

# Start backend on port 8000
cd src/backend
fastapi dev app/main.py --host localhost --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend on port 5000
cd ../frontend
npm run dev &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
