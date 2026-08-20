import { useState, type ComponentPropsWithoutRef, type CSSProperties } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api, type ChatQueryResponse, type ChunkOut, type CitationOut } from "../api/client";

const mdThStyle: CSSProperties = {
  textAlign: "left",
  borderBottom: "2px solid #ccc",
  padding: "0.4rem 0.6rem",
};
const mdTdStyle: CSSProperties = { borderBottom: "1px solid #eee", padding: "0.4rem 0.6rem" };

const markdownComponents = {
  table: (props: ComponentPropsWithoutRef<"table">) => (
    <table style={{ borderCollapse: "collapse", margin: "0.5rem 0" }} {...props} />
  ),
  th: (props: ComponentPropsWithoutRef<"th">) => <th style={mdThStyle} {...props} />,
  td: (props: ComponentPropsWithoutRef<"td">) => <td style={mdTdStyle} {...props} />,
  p: (props: ComponentPropsWithoutRef<"p">) => <p style={{ margin: "0.4rem 0" }} {...props} />,
};

interface ChatTurn {
  query: string;
  response: ChatQueryResponse;
}

const overlayStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.4)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};
const modalStyle: CSSProperties = {
  background: "white",
  color: "#1a1a1a",
  padding: "1.5rem",
  borderRadius: 8,
  maxWidth: 520,
  maxHeight: "80vh",
  overflowY: "auto",
};

export function ChatPage() {
  const [query, setQuery] = useState("");
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openFragment, setOpenFragment] = useState<ChunkOut | null>(null);

  const handleAsk = async () => {
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);
    try {
      const response = await api.chatQuery(query, conversationId);
      setConversationId(response.conversation_id);
      setTurns((prev) => [...prev, { query, response }]);
      setQuery("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  const handleCitationClick = async (citation: CitationOut) => {
    try {
      const chunk = await api.getChunk(citation.document_id, citation.chunk_id);
      setOpenFragment(chunk);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load source fragment");
    }
  };

  return (
    <div>
      <h2>Chat</h2>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem", marginBottom: "1rem" }}>
        {turns.map((turn, i) => (
          <div key={i} style={{ border: "1px solid #ddd", borderRadius: 8, padding: "0.75rem" }}>
            <p>
              <strong>Q:</strong> {turn.query}
            </p>
            <div>
              <strong>A:</strong>
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {turn.response.answer}
              </ReactMarkdown>
            </div>
            {turn.response.citations.length > 0 && (
              <div>
                <strong>Sources:</strong>
                <ul>
                  {turn.response.citations.map((c) => (
                    <li key={c.chunk_id}>
                      <button onClick={() => handleCitationClick(c)}>
                        {c.document_title}
                        {c.page_number ? ` (page ${c.page_number})` : ""}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {turns.length === 0 && <p>Ask a question about your uploaded documents.</p>}
      </div>

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      <div style={{ display: "flex", gap: "0.5rem" }}>
        <input
          style={{ flex: 1 }}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="Ask a question about your documents..."
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </div>

      {openFragment && (
        <div style={overlayStyle} onClick={() => setOpenFragment(null)}>
          <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
            <h3>{openFragment.document_title}</h3>
            <p style={{ fontSize: "0.85rem", color: "#666" }}>
              {openFragment.page_number ? `Page ${openFragment.page_number}` : "No page info"}
              {openFragment.section_title ? ` · ${openFragment.section_title}` : ""}
            </p>
            <p>{openFragment.content}</p>
            <button onClick={() => setOpenFragment(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
