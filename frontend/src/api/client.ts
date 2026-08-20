const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const TOKEN_STORAGE_KEY = "atlas_token";

let authToken: string | null = localStorage.getItem(TOKEN_STORAGE_KEY);

export function setAuthToken(token: string | null): void {
  authToken = token;
  if (token) {
    localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

export function getAuthToken(): string | null {
  return authToken;
}

export interface UserOut {
  id: string;
  email: string;
  role: string;
}

export interface DocumentOut {
  id: string;
  title: string;
  mime_type: string;
  status: string;
  created_at: string;
}

export interface DocumentStatusOut {
  id: string;
  status: string;
  error_message: string | null;
}

export interface ChunkOut {
  id: string;
  document_id: string;
  document_title: string;
  version_number: number;
  chunk_index: number;
  content: string;
  page_number: number | null;
  section_title: string | null;
}

export interface CitationOut {
  chunk_id: string;
  document_id: string;
  document_title: string;
  version_number: number;
  page_number: number | null;
  section_title: string | null;
  score: number;
}

export interface ChatQueryResponse {
  conversation_id: string;
  message_id: string;
  answer: string;
  citations: CitationOut[];
}

export interface EndpointLatencyOut {
  endpoint: string;
  count: number;
  p50_total_ms: number;
  p95_total_ms: number;
}

export interface RecentRequestOut {
  endpoint: string;
  total_ms: number;
  stage_latencies_ms: Record<string, number>;
  created_at: string;
}

export interface MetricsOut {
  endpoints: EndpointLatencyOut[];
  recent: RecentRequestOut[];
}

export interface EvaluationRunOut {
  id: string;
  name: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  result_count: number;
  mean_metrics: Record<string, number>;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers);
  if (authToken) {
    headers.set("Authorization", `Bearer ${authToken}`);
  }

  const response = await fetch(`${API_URL}${path}`, { ...options, headers });

  if (response.status === 401) {
    setAuthToken(null);
    window.dispatchEvent(new Event("atlas-unauthorized"));
  }
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Request failed (${response.status}): ${detail}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  register: (email: string, password: string) =>
    request<TokenResponse>("/api/v1/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) => {
    const form = new URLSearchParams();
    form.set("username", email);
    form.set("password", password);
    return request<TokenResponse>("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form.toString(),
    });
  },

  me: () => request<UserOut>("/api/v1/auth/me"),

  listDocuments: () => request<DocumentOut[]>("/api/v1/documents"),

  uploadDocument: (file: File, isPrivate: boolean) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("private", String(isPrivate));
    return request<DocumentOut>("/api/v1/documents", { method: "POST", body: formData });
  },

  getDocumentStatus: (id: string) =>
    request<DocumentStatusOut>(`/api/v1/documents/${id}/status`),

  deleteDocument: (id: string) =>
    request<void>(`/api/v1/documents/${id}`, { method: "DELETE" }),

  getChunk: (documentId: string, chunkId: string) =>
    request<ChunkOut>(`/api/v1/documents/${documentId}/chunks/${chunkId}`),

  chatQuery: (query: string, conversationId?: string) =>
    request<ChatQueryResponse>("/api/v1/chat/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, conversation_id: conversationId ?? null }),
    }),

  getAdminMetrics: () => request<MetricsOut>("/api/v1/admin/metrics"),

  getAdminEvaluations: () => request<EvaluationRunOut[]>("/api/v1/admin/evaluations"),
};
