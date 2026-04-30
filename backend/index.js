import express from "express";
import axios from "axios";
import cors from "cors";
import { getSession, saveSession } from "./sessionStore.js";

const app = express();

app.use(cors());
app.use(express.json());

const FLASK_URL = process.env.FLASK_URL || "http://127.0.0.1:8001/search";

app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

app.post("/api/recipe-search", async (req, res) => {
  const { sessionId, query, top_k, matchedKeywords } = req.body;

  if (!sessionId || !query || !query.trim()) {
    return res.status(400).json({ error: "sessionId and query is required" });
  }

  try {
    const existingSession = (await getSession(sessionId)) || {
      history: [],
    };
    const flaskRes = await axios.post(FLASK_URL, {
      session_id: sessionId,
      query,
      matchedKeywords,
      top_k: top_k || 5,
    });

    const data = res.json(flaskRes.data);

    const interaction = {
      prompt: query,
      matchedKeywords: data.matchedKeywords || [],
      recommendations: data.recommendations || [],
      createdAt: new Date().toISOString(),
    };

    const newSession = {
      ...existingSession,
      lastPrompt: query,
      lastMatchedKeywords: interaction.matchedKeywords,
      lastRecommendations: interaction.recommendations,
      history: [...existingSession.history, interaction],
    };

    await saveSession(sessionId, newSession);

    return res.json({
      sessionId,
      matchedKeywords: interaction.matchedKeywords,
      recommendations: interaction.recommendations,
    });
  } catch (err) {
    console.error("Error calling Flask:", err.message);

    if (err.response) {
      return res
        .status(err.response.status)
        .json({ error: err.response.data || "Flask error" });
    }

    return res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/api/graph-recipe-search", async (req, res) => {
  try {
    const { sessionId, query, matchedKeywords, top_k = 5 } = req.body;

    const existingSession = (await getSession(sessionId)) || {
      history: [],
    };

    const response = await fetch(`http://127.0.0.1:8001/graph-search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, query, top_k, matchedKeywords }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Graph search error from model server:", errText);
      return res
        .status(500)
        .json({ error: "Graph search failed", detail: errText });
    }

    const data = await response.json();

    const interaction = {
      prompt: query,
      matchedKeywords: data.keywords || [],
      recommendations: data.results || [],
      createdAt: new Date().toISOString(),
    };

    const newSession = {
      ...existingSession,
      lastPrompt: query,
      lastMatchedKeywords: interaction.matchedKeywords,
      lastRecommendations: interaction.recommendations,
      history: [...existingSession.history, interaction],
    };

    await saveSession(sessionId, newSession);

    return res.json({
      sessionId,
      matchedKeywords: interaction.matchedKeywords,
      recommendations: interaction.recommendations,
    });
  } catch (err) {
    console.error("Error calling graph model server:", err);
    res.status(500).json({ error: "Graph model server call failed" });
  }
});

app.post("/api/jiewan-recipe-search", async (req, res) => {
  try {
    const {
      sessionId,
      query,
      matchedKeywords,
      filterKeywords,
      top_k = 5,
    } = req.body;

    const existingSession = (await getSession(sessionId)) || {
      history: [],
    };

    const response = await fetch(`http://127.0.0.1:8001/jiewan-search-v2`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId,
        query,
        top_k,
        matchedKeywords,
        filterKeywords,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error("Graph search error from model server:", errText);
      return res
        .status(500)
        .json({ error: "Graph search failed", detail: errText });
    }

    const data = await response.json();

    const interaction = {
      prompt: query,
      matchedKeywords: data.keywords || [],
      recommendations: data.results || [],
      createdAt: new Date().toISOString(),
    };

    const newSession = {
      ...existingSession,
      lastPrompt: query,
      lastMatchedKeywords: interaction.matchedKeywords,
      lastRecommendations: interaction.recommendations,
      history: [...existingSession.history, interaction],
    };

    await saveSession(sessionId, newSession);

    return res.json({
      sessionId,
      matchedKeywords: interaction.matchedKeywords,
      recommendations: interaction.recommendations,
    });
  } catch (err) {
    console.error("Error calling graph model server:", err);
    res.status(500).json({ error: "Graph model server call failed" });
  }
});

app.get("/api/recipe/:id", async (req, res) => {
  const { id } = req.params;

  if (!id) {
    return res.status(400).json({ error: "recipe id is required" });
  }

  try {
    const flaskRes = await fetch(`http://127.0.0.1:8001/crawl-recipe/${id}`);

    if (!flaskRes.ok) {
      const text = await flaskRes.text();
      console.error("Flask crawl error:", text);
      return res
        .status(500)
        .json({ error: "model server error", detail: text });
    }

    const data = await flaskRes.json();

    return res.json(data);
  } catch (err) {
    console.error("Error calling Flask /crawl-recipe:", err);
    return res.status(500).json({ error: "crawl request failed" });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Express server listening on port ${PORT}`);
});
