// src/context/SessionContext.js
import React, {
  createContext,
  useContext,
  useState,
  useMemo,
  useEffect,
} from "react";

const SessionContext = createContext(null);

// localStorage key
const STORAGE_KEY = "sessionState";

export function SessionProvider({ children }) {
  const [sessionId, setSessionId] = useState(null);
  const [currentPrompt, setCurrentPrompt] = useState("");
  const [promptHistory, setPromptHistory] = useState([]);
  const [matchedKeywords, setMatchedKeywords] = useState({});
  const [recommendations, setRecommendations] = useState([]);

  // ⭐ 1. 세션 초기화
  const resetSession = () => {
    setCurrentPrompt("");
    setPromptHistory([]);
    setMatchedKeywords({});
    setRecommendations([]);

    // localStorage에서도 삭제
    window.localStorage.removeItem(STORAGE_KEY);
  };

  // ⭐ 2. 새 interaction 추가 → promptHistory 업데이트
  const addInteraction = ({ prompt, matchedKeywords, recommendations }) => {
    const interaction = {
      prompt,
      matchedKeywords,
      recommendations,
      createdAt: new Date().toISOString(),
    };

    setPromptHistory((prev) => [...prev, interaction]);
    setMatchedKeywords(matchedKeywords || {});
    setRecommendations(recommendations || []);
  };

  // ⭐ 3. localStorage에서 세션 복원
  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);

    if (saved) {
      try {
        const parsed = JSON.parse(saved);

        if (parsed.sessionId) setSessionId(parsed.sessionId);
        if (parsed.currentPrompt) setCurrentPrompt(parsed.currentPrompt);
        if (Array.isArray(parsed.promptHistory))
          setPromptHistory(parsed.promptHistory);
        setMatchedKeywords(
          parsed.promptHistory[parsed.promptHistory.length - 1].matchedKeywords
        );
        // if (Array.isArray(parsed.matchedKeywords))
        //   setMatchedKeywords(parsed.matchedKeywords);
        if (Array.isArray(parsed.recommendations))
          setRecommendations(parsed.recommendations);
      } catch (e) {
        console.error("Failed to restore session from localStorage:", e);
      }
    }

    // sessionId 없으면 생성
    if (!saved) {
      const newId =
        crypto.randomUUID?.() ||
        `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      setSessionId(newId);

      // 저장
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          sessionId: newId,
          currentPrompt: "",
          promptHistory: [],
          matchedKeywords: {},
          recommendations: [],
        })
      );
    }
  }, []);

  // ⭐ 4. 모든 상태가 바뀔 때마다 localStorage에 자동 저장
  useEffect(() => {
    if (!sessionId) return;

    const state = {
      sessionId,
      currentPrompt,
      promptHistory,
      matchedKeywords,
      recommendations,
    };

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [
    sessionId,
    currentPrompt,
    promptHistory,
    matchedKeywords,
    recommendations,
  ]);

  const value = useMemo(
    () => ({
      sessionId,
      currentPrompt,
      setCurrentPrompt,
      promptHistory,
      matchedKeywords,
      recommendations,
      resetSession,
      addInteraction,
    }),
    [sessionId, currentPrompt, promptHistory, matchedKeywords, recommendations]
  );

  return (
    <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
  );
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) {
    throw new Error("useSession must be used within a SessionProvider");
  }
  return ctx;
}
