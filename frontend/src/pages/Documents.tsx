import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";

import { api, type DocumentOut } from "../api/client";

const STATUS_COLORS: Record<string, string> = {
  ready: "#1a7f37",
  processing: "#9a6700",
  pending: "#9a6700",
  failed: "#cf222e",
};

const thStyle: CSSProperties = {
  textAlign: "left",
  borderBottom: "1px solid #ddd",
  padding: "0.5rem",
};
const tdStyle: CSSProperties = { borderBottom: "1px solid #eee", padding: "0.5rem" };

interface DocumentsPageProps {
  isAdmin: boolean;
}

export function DocumentsPage({ isAdmin }: DocumentsPageProps) {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [uploading, setUploading] = useState(false);
  const [isPrivate, setIsPrivate] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDocuments = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  useEffect(() => {
    const hasPending = documents.some(
      (d) => d.status === "processing" || d.status === "pending",
    );
    if (!hasPending) return;
    const interval = setInterval(loadDocuments, 2000);
    return () => clearInterval(interval);
  }, [documents, loadDocuments]);

  const handleUpload = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      await api.uploadDocument(file, isPrivate);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setIsPrivate(false);
      await loadDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    await api.deleteDocument(id);
    await loadDocuments();
  };

  return (
    <div>
      <h2>Documents</h2>

      {isAdmin ? (
        <div style={{ marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <input ref={fileInputRef} type="file" accept=".pdf,.txt,.docx" />
          <label style={{ fontSize: "0.9rem" }}>
            <input
              type="checkbox"
              checked={isPrivate}
              onChange={(e) => setIsPrivate(e.target.checked)}
            />{" "}
            Private (only visible to you)
          </label>
          <button onClick={handleUpload} disabled={uploading}>
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>
      ) : (
        <p style={{ color: "#666", fontSize: "0.9rem" }}>
          Only admins can upload documents. You can query and browse documents shared with you.
        </p>
      )}

      {error && <p style={{ color: "crimson" }}>{error}</p>}

      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={thStyle}>Title</th>
            <th style={thStyle}>Status</th>
            <th style={thStyle}>Uploaded</th>
            {isAdmin && <th style={thStyle} />}
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.id}>
              <td style={tdStyle}>{doc.title}</td>
              <td style={tdStyle}>
                <span style={{ color: STATUS_COLORS[doc.status] ?? "#333", fontWeight: 600 }}>
                  {doc.status}
                </span>
              </td>
              <td style={tdStyle}>{new Date(doc.created_at).toLocaleString()}</td>
              {isAdmin && (
                <td style={tdStyle}>
                  <button onClick={() => handleDelete(doc.id)}>Delete</button>
                </td>
              )}
            </tr>
          ))}
          {documents.length === 0 && (
            <tr>
              <td style={tdStyle} colSpan={isAdmin ? 4 : 3}>
                No documents visible to you yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
