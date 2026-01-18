/**
 * Type definitions for RLM-Mem MCP Server
 * Implements types for Recursive Language Model context management
 */

import { z } from "zod";

// ============================================================================
// Document & Chunk Types
// ============================================================================

export interface DocumentChunk {
  id: string;
  content: string;
  tokenCount: number;
  metadata: ChunkMetadata;
  embedding?: number[];
}

export interface ChunkMetadata {
  documentId: string;
  documentTitle?: string;
  chunkIndex: number;
  totalChunks: number;
  startOffset: number;
  endOffset: number;
  sectionTitle?: string;
  depth: number; // RLM recursion depth
}

export interface Document {
  id: string;
  title: string;
  content: string;
  tokenCount: number;
  chunks: DocumentChunk[];
  summary?: string;
  metadata: DocumentMetadata;
}

export interface DocumentMetadata {
  source: string;
  createdAt: Date;
  updatedAt: Date;
  contentType: "text" | "markdown" | "code" | "json";
  tags?: string[];
}

// ============================================================================
// RLM Processing Types
// ============================================================================

export interface RLMNode {
  id: string;
  depth: number;
  parentId?: string;
  childIds: string[];
  content: string;
  summary?: string;
  tokenCount: number;
  processed: boolean;
}

export interface RLMTree {
  rootId: string;
  nodes: Map<string, RLMNode>;
  maxDepth: number;
  documentId: string;
}

export interface RLMQueryResult {
  query: string;
  nodeId: string;
  depth: number;
  response: string;
  relevanceScore: number;
  tokenCount: number;
}

export interface RLMAggregatedResult {
  originalQuery: string;
  subResults: RLMQueryResult[];
  finalResponse: string;
  totalTokensUsed: number;
  cacheHits: number;
  processingTime: number;
}

// ============================================================================
// Cache Types (Anthropic Prompt Caching)
// ============================================================================

export interface CacheEntry {
  id: string;
  content: string;
  tokenCount: number;
  createdAt: Date;
  lastAccessedAt: Date;
  hitCount: number;
  ttl: "5m" | "1h";
}

export interface CacheConfig {
  enabled: boolean;
  defaultTtl: "5m" | "1h";
  maxEntries: number;
  minTokensForCache: number; // Minimum 1024 for Sonnet, 4096 for Opus
}

export interface CacheStats {
  totalEntries: number;
  totalTokensCached: number;
  cacheHits: number;
  cacheMisses: number;
  estimatedSavings: number; // In tokens
}

// ============================================================================
// Memory Store Types
// ============================================================================

export interface MemoryEntry {
  id: string;
  key: string;
  value: string;
  summary?: string;
  tokenCount: number;
  importance: number; // 0-1 scale
  createdAt: Date;
  lastAccessedAt: Date;
  accessCount: number;
  tags: string[];
}

export interface MemoryStore {
  entries: Map<string, MemoryEntry>;
  totalTokens: number;
  maxTokens: number;
}

export interface MemorySearchResult {
  entry: MemoryEntry;
  relevanceScore: number;
}

// ============================================================================
// MCP Tool Input/Output Schemas
// ============================================================================

export const IngestDocumentSchema = z.object({
  content: z.string().describe("The document content to ingest"),
  title: z.string().optional().describe("Optional title for the document"),
  contentType: z
    .enum(["text", "markdown", "code", "json"])
    .default("text")
    .describe("Type of content"),
  tags: z.array(z.string()).optional().describe("Optional tags for categorization"),
});

export const QueryContextSchema = z.object({
  query: z.string().describe("The query to search for in the context"),
  maxDepth: z
    .number()
    .min(1)
    .max(5)
    .default(2)
    .describe("Maximum RLM recursion depth"),
  maxTokens: z
    .number()
    .min(100)
    .max(100000)
    .default(4000)
    .describe("Maximum tokens in response"),
  useCache: z.boolean().default(true).describe("Whether to use prompt caching"),
});

export const StoreMemorySchema = z.object({
  key: z.string().describe("Unique key for this memory"),
  value: z.string().describe("The content to store"),
  importance: z
    .number()
    .min(0)
    .max(1)
    .default(0.5)
    .describe("Importance score (0-1)"),
  tags: z.array(z.string()).optional().describe("Tags for categorization"),
});

export const RecallMemorySchema = z.object({
  query: z.string().describe("Search query for memory recall"),
  limit: z.number().min(1).max(20).default(5).describe("Maximum results to return"),
  minImportance: z
    .number()
    .min(0)
    .max(1)
    .default(0)
    .describe("Minimum importance threshold"),
});

export const SummarizeContextSchema = z.object({
  documentId: z.string().optional().describe("Specific document to summarize"),
  maxLength: z
    .number()
    .min(100)
    .max(10000)
    .default(1000)
    .describe("Maximum summary length in tokens"),
});

export const GetContextStatsSchema = z.object({
  includeDocuments: z.boolean().default(true),
  includeMemory: z.boolean().default(true),
  includeCache: z.boolean().default(true),
});

// ============================================================================
// Configuration Types
// ============================================================================

export interface RLMConfig {
  maxChunkTokens: number;
  overlapTokens: number;
  maxRecursionDepth: number;
  summaryMaxTokens: number;
  minChunkTokens: number;
}

export interface ServerConfig {
  rlm: RLMConfig;
  cache: CacheConfig;
  memory: {
    maxTokens: number;
    decayFactor: number; // For importance decay over time
  };
  anthropic: {
    model: string;
    maxTokens: number;
    temperature: number;
  };
}

// Type exports for Zod schemas
export type IngestDocumentInput = z.infer<typeof IngestDocumentSchema>;
export type QueryContextInput = z.infer<typeof QueryContextSchema>;
export type StoreMemoryInput = z.infer<typeof StoreMemorySchema>;
export type RecallMemoryInput = z.infer<typeof RecallMemorySchema>;
export type SummarizeContextInput = z.infer<typeof SummarizeContextSchema>;
export type GetContextStatsInput = z.infer<typeof GetContextStatsSchema>;
