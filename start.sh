#!/bin/bash
# Start frontend on port 8080
python -m http.server 8080 -d frontend &

# Start backend on port 8000
cd backend
python main.py
