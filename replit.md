# Board Matcher - Replit Setup

## Overview
Board Matcher is a TF-IDF-based connection matching application that helps find professional connections based on employment history and board service data. The application uses machine learning to match people based on their professional backgrounds.

**Date Created:** November 3, 2025

## Project Architecture

### Technology Stack
- **Backend:** FastAPI (Python 3.11)
  - TF-IDF model using scikit-learn
  - Pandas for data processing
  - Excel file parsing with openpyxl
- **Frontend:** React 19 + TypeScript + Vite
  - Tailwind CSS for styling
  - Lucide React for icons

### Project Structure
```
.
├── src/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py          # FastAPI application
│   │   │   ├── models.py        # Pydantic models
│   │   │   ├── tfidf_model.py   # TF-IDF matching engine
│   │   │   └── data_parser.py   # Excel data parser
│   │   └── requirements.txt
│   └── frontend/
│       ├── src/
│       │   ├── App.tsx          # Main React component
│       │   └── main.tsx
│       ├── package.json
│       └── vite.config.ts       # Vite configuration
├── start.sh                      # Startup script for both services
└── replit.md                     # This file
```

## Configuration

### Backend
- **Port:** 8000 (localhost only)
- **Host:** localhost
- **CORS:** Enabled for all origins (configured for development)

### Frontend
- **Port:** 5000 (required for Replit webview)
- **Host:** 0.0.0.0 (allows Replit proxy access)
- **API URL:** Set via VITE_API_URL environment variable in `src/frontend/.env`

### Environment Variables
- `VITE_API_URL`: Backend API URL (set in src/frontend/.env)
- `REPLIT_DEV_DOMAIN`: Automatically provided by Replit

## Running the Application

### Development
The application runs automatically via the "Board Matcher" workflow, which executes `start.sh`:
1. Starts FastAPI backend on localhost:8000
2. Starts Vite frontend on 0.0.0.0:5000
3. Frontend is accessible via Replit's webview

### Manual Start
```bash
bash start.sh
```

## Features
1. **Dataset Upload:** Upload Excel files with professional data (Name, Employment, Board Service)
2. **TF-IDF Matching:** Find matching connections based on professional background
3. **Export Results:** Download match results as Excel files
4. **Real-time Search:** Instant matching with configurable thresholds

## Dependencies

### Backend (Python)
- fastapi[standard]
- scikit-learn
- pandas
- numpy
- openpyxl

### Frontend (Node.js)
- React 19
- TypeScript
- Vite
- Tailwind CSS v4
- lucide-react

## Recent Changes
- **2025-11-03:** Initial Replit setup
  - Installed Python 3.11 and Node.js 20
  - Configured Vite to bind to 0.0.0.0:5000 for Replit webview
  - Created startup script to run backend and frontend together
  - Set up workflow for automatic startup
  - Added comprehensive .gitignore

## Notes
- The backend uses an in-memory dataset that must be uploaded via the UI
- The TF-IDF model is initialized when a dataset is uploaded
- Frontend communicates with backend via the Replit domain proxy
