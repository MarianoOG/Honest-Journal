# Honest Journal - Personal Growth App

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This repository is an app to a systematic approach to understanding life, making better decisions, and continuously improving. It represents my journey of personal growth and philosophical exploration.

## Core Components

1. **Journal**
   - **Thoughts**: Capture thoughts based on questions that you create and challenge your assumptions, blind-spots and contradictions.
   - **Memories**: Free text entries about anecdotes, experiences and things you're thankful for and extract insights from them.
   - **Learning**: Capture your learnings from books, articles, videos, and podcasts and see how they connect to your life.
   
2. **Review System**
   - **Challenge Beliefs**: LLM-assisted questioning of your assumptions, blind-spots and contradictions.
   - **Generate Insights**: Generate unique perspectives on your thoughts based on your entries.
   - **Summarize and Consolidate**: Come back to a compacted version of your entries that you can consult and expand on.
   
3. **Visualization and Analysis**
   - **Mood Tracking**: Track your emotional state over time with data visualization.
   - **Resource Library**: Curated content on personal development and reflection techniques.
   - **Integration with Other Tools**: Connect with meditation apps, fitness trackers, etc.

## Technical Stack

- **Backend**: FastAPI for API
- **Frontend**: Streamlit for interactive interfaces
- **Infrastructure**: GCP Cloud Run for containerized services
- **Storage**: SQLite for Reflexion Entries and Questions stored in GCP bucket
- **Third Party APIs**: OpenAI for LLM integration

## Getting Started

1. **Prerequisites**
   - OpenAI API key
   - Bucket in GCP
   - GCP credentials

2. **Configuration**

You first need to create a [`.env`](.env) file with the following variables:
```
OPENAI_API_KEY=your_openai_key
BUCKET_NAME=your_google_cloud_service_bucket
DATABASE_URL=your_sqlite_database_url
```

3. **Local Development**

   You can run the services locally using the following commands:
   ```bash
   # Install dependencies
   pip install -r backend/requirements.txt -r frontend/requirements.txt

   # Run services
   uvicorn backend/fastapi_app:app --reload --port 8000
   streamlit run frontend/main.py --server.port 8080
   ```
   
   You can also use docker compose to run the services
   ```bash
   docker compose up --build
   ```

4. **Deployment**
   - Automated deployment is configured via GitHub Actions
   - Workflow configuration is in [.github/workflows/deploy.yml](.github/workflows/deploy.yml)
   - Requires GCP credentials configured as repository secrets

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
