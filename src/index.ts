/**
 * RLM-Mem MCP Server (TypeScript Implementation)
 *
 * An MCP server implementing the Recursive Language Model (RLM) technique
 * for ultimate context management with Claude Code.
 *
 * This TypeScript implementation provides an alternative to the Python server,
 * with the same functionality but using the Node.js ecosystem.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { RLMContextManager } from "./core/rlm-context-manager.js";
import {
  IngestDocumentSchema,
  QueryContextSchema,
  StoreMemorySchema,
  RecallMemorySchema,
  GetContextStatsSchema,
} from "./types/index.js";
import { countTokens, getTokenStats } from "./utils/tokenizer.js";

// Configuration from environment
const config = {
  apiKey: process.env.ANTHROPIC_API_KEY || "",
  model: process.env.RLM_MODEL || "claude-sonnet-4-5-20241022",
  maxResultTokens: parseInt(process.env.RLM_MAX_RESULT_TOKENS || "4000"),
  useCache: process.env.RLM_USE_CACHE !== "false",
  cacheTtl: (process.env.RLM_CACHE_TTL || "5m") as "5m" | "1h",
};

// Memory store for persistence
const memoryStore = new Map<
  string,
  { value: string; tags: string[]; tokenCount: number }
>();

// RLM context manager instance
let contextManager: RLMContextManager | null = null;

function getContextManager(): RLMContextManager {
  if (!contextManager) {
    if (!config.apiKey) {
      throw new Error("ANTHROPIC_API_KEY environment variable is required");
    }
    contextManager = new RLMContextManager(config.apiKey, {
      maxChunkTokens: 8000,
      overlapTokens: 200,
      maxRecursionDepth: 3,
      summaryMaxTokens: 500,
    });
  }
  return contextManager;
}

async function main() {
  const server = new Server(
    {
      name: "rlm-recursive-memory",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // List available tools
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: "rlm_ingest",
        description:
          "Ingest a document or code content into the RLM context system for recursive processing. " +
          "Use this to add large documents that need to be queried later.",
        inputSchema: {
          type: "object",
          properties: {
            content: {
              type: "string",
              description: "The document content to ingest",
            },
            title: {
              type: "string",
              description: "Optional title for the document",
            },
            contentType: {
              type: "string",
              enum: ["text", "markdown", "code", "json"],
              default: "text",
              description: "Type of content",
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Optional tags for categorization",
            },
          },
          required: ["content"],
        },
      },
      {
        name: "rlm_query",
        description:
          "Query the ingested context using RLM recursive processing. " +
          "This processes large documents efficiently by chunking and aggregating results.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The query to search for in the context",
            },
            maxDepth: {
              type: "number",
              minimum: 1,
              maximum: 5,
              default: 2,
              description: "Maximum RLM recursion depth",
            },
            maxTokens: {
              type: "number",
              minimum: 100,
              maximum: 100000,
              default: 4000,
              description: "Maximum tokens in response",
            },
            useCache: {
              type: "boolean",
              default: true,
              description: "Whether to use prompt caching",
            },
          },
          required: ["query"],
        },
      },
      {
        name: "rlm_memory_store",
        description:
          "Store important information for later recall. " +
          "Use this to persist key findings, summaries, or context.",
        inputSchema: {
          type: "object",
          properties: {
            key: {
              type: "string",
              description: "Unique key for this memory",
            },
            value: {
              type: "string",
              description: "The content to store",
            },
            tags: {
              type: "array",
              items: { type: "string" },
              description: "Tags for categorization",
            },
          },
          required: ["key", "value"],
        },
      },
      {
        name: "rlm_memory_recall",
        description: "Recall stored information by key or search by tags.",
        inputSchema: {
          type: "object",
          properties: {
            key: {
              type: "string",
              description: "Exact key to recall",
            },
            searchTags: {
              type: "array",
              items: { type: "string" },
              description: "Tags to search for",
            },
          },
        },
      },
      {
        name: "rlm_status",
        description: "Get status of the RLM context system including statistics.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  }));

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      switch (name) {
        case "rlm_ingest": {
          const parsed = IngestDocumentSchema.parse(args);
          const manager = getContextManager();

          const document = await manager.ingestDocument(
            parsed.content,
            parsed.title,
            parsed.contentType,
            parsed.tags
          );

          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(
                  {
                    success: true,
                    documentId: document.id,
                    title: document.title,
                    tokenCount: document.tokenCount,
                    chunkCount: document.chunks.length,
                    summary: document.summary?.slice(0, 200) + "...",
                  },
                  null,
                  2
                ),
              },
            ],
          };
        }

        case "rlm_query": {
          const parsed = QueryContextSchema.parse(args);
          const manager = getContextManager();

          const result = await manager.queryContext(parsed.query, {
            maxDepth: parsed.maxDepth,
            maxTokens: parsed.maxTokens,
            useCache: parsed.useCache,
          });

          // Format as structured response
          const output = [
            "## RLM Query Result",
            "",
            `**Query:** ${result.originalQuery}`,
            `**Processing Time:** ${result.processingTime}ms`,
            `**Cache Hits:** ${result.cacheHits}`,
            `**Tokens Used:** ${result.totalTokensUsed}`,
            "",
            "### Response",
            "",
            result.finalResponse,
          ].join("\n");

          return {
            content: [{ type: "text", text: output }],
          };
        }

        case "rlm_memory_store": {
          const parsed = StoreMemorySchema.parse(args);

          memoryStore.set(parsed.key, {
            value: parsed.value,
            tags: parsed.tags || [],
            tokenCount: countTokens(parsed.value),
          });

          return {
            content: [
              {
                type: "text",
                text: `Stored memory with key '${parsed.key}' (${parsed.value.length} chars)`,
              },
            ],
          };
        }

        case "rlm_memory_recall": {
          const parsed = RecallMemorySchema.parse(args);
          const results: Array<{
            key: string;
            value: string;
            tags: string[];
          }> = [];

          if (parsed.query) {
            // Search by key (treat query as key for exact match)
            const entry = memoryStore.get(parsed.query);
            if (entry) {
              results.push({
                key: parsed.query,
                value: entry.value,
                tags: entry.tags,
              });
            }
          }

          // If no results from key search, search all entries
          if (results.length === 0) {
            for (const [key, entry] of memoryStore.entries()) {
              // Simple keyword search in value
              if (
                parsed.query &&
                entry.value.toLowerCase().includes(parsed.query.toLowerCase())
              ) {
                results.push({ key, value: entry.value, tags: entry.tags });
              }
            }
          }

          if (results.length === 0) {
            return {
              content: [{ type: "text", text: "No matching memories found" }],
            };
          }

          return {
            content: [
              { type: "text", text: JSON.stringify(results, null, 2) },
            ],
          };
        }

        case "rlm_status": {
          const manager = getContextManager();
          const stats = manager.getStats();

          const status = {
            server: {
              name: "rlm-recursive-memory",
              version: "1.0.0",
              runtime: "node",
            },
            configuration: {
              model: config.model,
              apiKeySet: !!config.apiKey,
              maxResultTokens: config.maxResultTokens,
              useCache: config.useCache,
              cacheTtl: config.cacheTtl,
            },
            context: stats,
            memory: {
              entries: memoryStore.size,
              totalTokens: Array.from(memoryStore.values()).reduce(
                (sum, e) => sum + e.tokenCount,
                0
              ),
            },
          };

          return {
            content: [{ type: "text", text: JSON.stringify(status, null, 2) }],
          };
        }

        default:
          return {
            content: [{ type: "text", text: `Unknown tool: ${name}` }],
            isError: true,
          };
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        content: [{ type: "text", text: `Error: ${errorMessage}` }],
        isError: true,
      };
    }
  });

  // Start server
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("RLM-Mem MCP Server (TypeScript) running on stdio");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
