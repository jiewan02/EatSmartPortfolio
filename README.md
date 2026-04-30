# EatSmart (해먹고말지) - AI Recipe Recommendation Service

EatSmart is a context-aware recipe recommendation system that uses Graph RAG, OpenAI embeddings, and a Neo4j knowledge graph. You type a natural-language prompt like "I have leftover pork and I'm stressed, what should I eat?" and the system recommends Korean recipes that best match your situation, ingredients, and preferences.

---

## How it works

The system has three layers that communicate with each other:

- **React frontend** (port 3000) - where the user types prompts and sees results
- **Express backend** (port 5000) - proxies requests between the frontend and model server, and stores session history in Redis
- **Flask model server** (port 8001) - extracts keywords from the prompt using a local LLM, runs a Graph RAG search over Neo4j, and returns ranked recipe results

| Folder | What it does |
|---|---|
| `client/` | React UI with search bar, results list, and recipe detail page |
| `backend/` | Express server for session management via Redis and routing to Flask |
| `model-server/` | Flask server for keyword extraction, graph search, and recipe crawling |
| `data-pipeline/` | One-time scripts to preprocess data, build the Neo4j graph, and generate embeddings |

---

## Prerequisites

Make sure the following are installed and available before you start:

| Tool | Version | Purpose |
|---|---|---|
| Node.js | v16 or higher | Running the backend and frontend |
| Python with conda | 3.10 or higher | Running the model server |
| Neo4j | Any recent version | Graph database, runs on port 7687 |
| Redis | Any recent version | Session storage, runs on port 6379 |
| OpenAI API key | - | Embedding generation and recipe tagging |
| Qwen2.5-14B-Instruct | - | Local LLM for keyword extraction |

**GPU note:** The keyword extractor loads a 14B-parameter model in 4-bit quantization using bitsandbytes. This requires a CUDA-capable GPU with at least 20 GB of VRAM. The model is loaded onto GPU device index 2 by default (`device_map={"": 2}` in `new_extractor_model.py`). If your setup is different, adjust that value before starting the server.

---

## Step 1 - Clone the repo and set up credentials

```bash
git clone <your-repo-url>
cd EatSmart_Clean
```

Copy the environment variable template and fill it in:

```bash
cp model-server/.env.example model-server/.env
```

Open `model-server/.env` and set all five values:

```
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
MODEL_PATH=/absolute/path/to/Qwen2.5-14B-Instruct
```

---

## Step 2 - Data pipeline (one-time setup)

Skip this entire step if you already have `dataset_preprocessed.csv` and `recipe_embeddings.npy` and can place them directly into `model-server/`. Then go straight to Step 3.

### 2a - Install pipeline dependencies

```bash
conda create -n recipe-pipeline python=3.10 -y
conda activate recipe-pipeline
pip install openai pandas tqdm python-dotenv neo4j numpy
```

Then copy your credentials so the pipeline scripts can read them:

```bash
cp model-server/.env data-pipeline/.env
```

### 2b - Preprocess and tag the raw dataset

Place your raw dataset file (`dataset_part2.csv`) inside `data-pipeline/`, then run:

```bash
cd data-pipeline
python data_preprocessing_safety.py
```

This calls the OpenAI API to tag each recipe with health, weather, and menu-style labels. It may take several minutes and will use OpenAI API credits.

### 2c - Build the Neo4j knowledge graph

Place `dataset_preprocessed.csv` (your fully cleaned and tagged recipe dataset) inside `data-pipeline/`, then run:

```bash
python build_graph.py
```

Once that finishes, copy the CSV to `model-server/` since the Flask server needs it at runtime:

```bash
cp dataset_preprocessed.csv ../model-server/dataset_preprocessed.csv
```

---

## Step 3 - Set up the model server environment

This step installs the Python dependencies for the Flask server. You need to complete this before generating embeddings (Step 4) and before starting the server (Step 5).

```bash
cd model-server
conda create -n recipe-model python=3.10 -y
conda activate recipe-model
pip install -r requirements.txt
```

---

## Step 4 - Generate recipe embeddings (skip if you already have recipe_embeddings.npy)

Make sure you completed Step 3 first and that `dataset_preprocessed.csv` is inside `model-server/`. Then run:

```bash
cd model-server
conda activate recipe-model
python ../data-pipeline/generate_embeddings.py
```

This reads `dataset_preprocessed.csv`, calls the OpenAI embeddings API for each recipe name, and saves the result as `recipe_embeddings.npy` inside `model-server/`. This file is not committed to git.

---

## Step 5 - Start the model server

Make sure both `dataset_preprocessed.csv` and `recipe_embeddings.npy` are inside `model-server/` before running this.

```bash
cd model-server
conda activate recipe-model
python app.py
```

The server runs on `http://127.0.0.1:8001`. On startup it loads the two data files and then loads the Qwen model from disk, which takes a few minutes. Wait until you see `Running on http://0.0.0.0:8001` in the terminal before moving on.

---

## Step 6 - Start the backend

Open a new terminal tab from the project root:

```bash
cd backend
npm install
npm start
```

The server runs on `http://localhost:5000`.

---

## Step 7 - Start the frontend

Open another new terminal tab from the project root:

```bash
cd client
npm install
npm start
```

This opens the app at `http://localhost:3000` in your browser automatically.

---

## Startup checklist

All five of the following must be running before the app will work:

| Service | How to start | Port |
|---|---|---|
| Neo4j | Neo4j Desktop, start the database | 7687 |
| Redis | `brew services start redis` on macOS, or `systemctl start redis` on Linux | 6379 |
| Flask model server | `python app.py` inside `model-server/` (Step 5) | 8001 |
| Express backend | `npm start` inside `backend/` (Step 6) | 5000 |
| React frontend | `npm start` inside `client/` (Step 7) | 3000 |

To confirm the backend is up, run `curl http://localhost:5000/api/health` in a terminal. It should return `{"status":"ok"}`.

---

## Folder structure

```
EatSmart_Clean/
    client/                      React frontend
        public/                  Static assets
        src/
            Pages/
                Landing/         Splash and intro page
                Home/            Search page and results list
                Recipe/          Recipe detail page
            Assets/              Shared components and images
            Context/             Session context provider

    backend/
        index.js                 Express routes
        redisClient.js           Redis connection helper
        sessionStore.js          Session read and write helpers

    model-server/
        app.py                   Flask entry point and routes
        jiewan_model_v2.py       Graph RAG search logic
        graph_similarity_v2.py   Neo4j similar-recipe queries
        new_extractor_model.py   Qwen LLM keyword extractor
        requirements.txt
        .env.example
        dataset_preprocessed.csv   (not in git, add manually or generate via Step 2)
        recipe_embeddings.npy      (not in git, generated by Step 4)

    data-pipeline/
        data_preprocessing_safety.py   Tag raw recipes via GPT
        build_graph.py                 Populate the Neo4j graph
        generate_embeddings.py         Generate the .npy embeddings file
```

---

## Technology stack

| Area | Tools |
|---|---|
| Frontend | React 19, React Router 7, CSS Modules |
| Backend | Node.js, Express 5, Redis |
| Model server | Python 3.10, Flask, OpenAI SDK |
| AI and search | OpenAI gpt-4o-mini for keyword tagging, text-embedding-3-small for embeddings, Qwen2.5-14B-Instruct for keyword extraction, Graph RAG |
| Database | Neo4j for the recipe graph, Redis for sessions |
