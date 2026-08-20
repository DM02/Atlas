import { useEffect, useState } from "react";

import { api, getAuthToken, setAuthToken, type UserOut } from "./api/client";
import { AdminPage } from "./pages/Admin";
import { ChatPage } from "./pages/Chat";
import { DocumentsPage } from "./pages/Documents";
import { LoginPage } from "./pages/Login";

type Tab = "chat" | "documents" | "admin";

function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [currentUser, setCurrentUser] = useState<UserOut | null>(null);
  const [checkingAuth, setCheckingAuth] = useState(true);

  const loadCurrentUser = async () => {
    if (!getAuthToken()) {
      setCurrentUser(null);
      setCheckingAuth(false);
      return;
    }
    try {
      setCurrentUser(await api.me());
    } catch {
      setAuthToken(null);
      setCurrentUser(null);
    } finally {
      setCheckingAuth(false);
    }
  };

  useEffect(() => {
    loadCurrentUser();
  }, []);

  useEffect(() => {
    const handleUnauthorized = () => setCurrentUser(null);
    window.addEventListener("atlas-unauthorized", handleUnauthorized);
    return () => window.removeEventListener("atlas-unauthorized", handleUnauthorized);
  }, []);

  const handleLogout = () => {
    setAuthToken(null);
    setCurrentUser(null);
  };

  if (checkingAuth) {
    return null;
  }

  if (!currentUser) {
    return (
      <main style={{ fontFamily: "sans-serif", padding: "2rem", maxWidth: 800, margin: "0 auto" }}>
        <h1>Atlas</h1>
        <p style={{ color: "#666" }}>
          Production-oriented RAG platform — retrieval quality, evaluation, observability and
          secure document access.
        </p>
        <LoginPage onAuthenticated={loadCurrentUser} />
      </main>
    );
  }

  return (
    <main style={{ fontFamily: "sans-serif", padding: "2rem", maxWidth: 800, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h1>Atlas</h1>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span style={{ color: "#666", fontSize: "0.9rem" }}>
            {currentUser.email} ({currentUser.role})
          </span>
          <button onClick={handleLogout}>Log out</button>
        </div>
      </div>
      <p style={{ color: "#666" }}>
        Production-oriented RAG platform — retrieval quality, evaluation, observability and
        secure document access.
      </p>

      <nav style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem" }}>
        <button onClick={() => setTab("chat")} disabled={tab === "chat"}>
          Chat
        </button>
        <button onClick={() => setTab("documents")} disabled={tab === "documents"}>
          Documents
        </button>
        {currentUser.role === "admin" && (
          <button onClick={() => setTab("admin")} disabled={tab === "admin"}>
            Admin
          </button>
        )}
      </nav>

      {tab === "chat" && <ChatPage />}
      {tab === "documents" && <DocumentsPage isAdmin={currentUser.role === "admin"} />}
      {tab === "admin" && currentUser.role === "admin" && <AdminPage />}
    </main>
  );
}

export default App;
