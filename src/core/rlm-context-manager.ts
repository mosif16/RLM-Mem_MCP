/**
 * RLM Context Manager
 * Implements the Recursive Language Model technique for long context handling
 *
 * Based on the RLM paper approach:
 * 1. Split documents into hierarchical chunks
 * 2. Process queries recursively through the tree
 * 3. Aggregate responses from sub-queries
 * 4. Maintain memory of key findings
 */

import Anthropic from "@anthropic-ai/sdk";
import { splitDocument } from "../utils/text-splitter.js";
import { countTokens, meetsMinCacheThreshold } from "../utils/tokenizer.js";
import type {
  Document,
  DocumentChunk,
  RLMNode,
  RLMTree,
  RLMQueryResult,
  RLMAggregatedResult,
  RLMConfig,
  CacheConfig,
} from "../types/index.js";

interface ProcessingContext {
  query: string;
  maxDepth: number;
  currentDepth: number;
  useCache: boolean;
  accumulatedResults: RLMQueryResult[];
  tokenBudget: number;
  tokensUsed: number;
  cacheHits: number;
}

export class RLMContextManager {
  private documents: Map<string, Document> = new Map();
  private trees: Map<string, RLMTree> = new Map();
  private client: Anthropic;
  private config: RLMConfig;
  private cacheConfig: CacheConfig;
  private model: string;

  constructor(
    apiKey: string,
    config: Partial<RLMConfig> = {},
    cacheConfig: Partial<CacheConfig> = {}
  ) {
    this.client = new Anthropic({ apiKey });

    // Default RLM configuration
    this.config = {
      maxChunkTokens: 4000,
      overlapTokens: 200,
      maxRecursionDepth: 3,
      summaryMaxTokens: 500,
      minChunkTokens: 100,
      ...config,
    };

    // Default cache configuration
    this.cacheConfig = {
      enabled: true,
      defaultTtl: "5m",
      maxEntries: 100,
      minTokensForCache: 1024,
      ...cacheConfig,
    };

    this.model = "claude-sonnet-4-5-20241022";
  }

  /**
   * Ingest a document and build the RLM tree structure
   */
  async ingestDocument(
    content: string,
    title?: string,
    contentType: "text" | "markdown" | "code" | "json" = "text",
    tags?: string[]
  ): Promise<Document> {
    const documentId = `doc-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    const tokenCount = countTokens(content);

    // Split document into chunks
    const chunks = splitDocument(content, documentId, {
      maxTokens: this.config.maxChunkTokens,
      overlapTokens: this.config.overlapTokens,
      respectBoundaries: true,
      contentType,
    });

    // Build RLM tree from chunks
    const tree = await this.buildRLMTree(documentId, chunks);
    this.trees.set(documentId, tree);

    // Generate document summary
    const summary = await this.generateSummary(content, this.config.summaryMaxTokens);

    const document: Document = {
      id: documentId,
      title: title || `Document ${documentId}`,
      content,
      tokenCount,
      chunks,
      summary,
      metadata: {
        source: "direct-input",
        createdAt: new Date(),
        updatedAt: new Date(),
        contentType,
        tags,
      },
    };

    this.documents.set(documentId, document);

    return document;
  }

  /**
   * Build a hierarchical RLM tree from document chunks
   */
  private async buildRLMTree(documentId: string, chunks: DocumentChunk[]): Promise<RLMTree> {
    const nodes = new Map<string, RLMNode>();

    // Create leaf nodes from chunks (depth 0)
    const leafNodes: RLMNode[] = chunks.map((chunk, index) => ({
      id: `${documentId}-node-${index}`,
      depth: 0,
      childIds: [],
      content: chunk.content,
      tokenCount: chunk.tokenCount,
      processed: false,
    }));

    leafNodes.forEach((node) => nodes.set(node.id, node));

    // Build hierarchical structure (bottom-up)
    let currentLevel = leafNodes;
    let depth = 1;

    while (currentLevel.length > 1 && depth <= this.config.maxRecursionDepth) {
      const nextLevel: RLMNode[] = [];
      const groupSize = Math.ceil(Math.sqrt(currentLevel.length)); // Adaptive grouping

      for (let i = 0; i < currentLevel.length; i += groupSize) {
        const children = currentLevel.slice(i, i + groupSize);
        const childIds = children.map((c) => c.id);

        // Generate summary for this group
        const combinedContent = children.map((c) => c.summary || c.content).join("\n\n---\n\n");

        const summary = await this.generateSummary(combinedContent, this.config.summaryMaxTokens);

        const parentNode: RLMNode = {
          id: `${documentId}-node-d${depth}-${i}`,
          depth,
          childIds,
          content: combinedContent,
          summary,
          tokenCount: countTokens(summary || combinedContent),
          processed: false,
        };

        // Update children with parent reference
        children.forEach((child) => {
          child.parentId = parentNode.id;
          nodes.set(child.id, child);
        });

        nextLevel.push(parentNode);
        nodes.set(parentNode.id, parentNode);
      }

      currentLevel = nextLevel;
      depth++;
    }

    // The last remaining node (or single root) becomes the root
    const rootId = currentLevel[0]?.id || leafNodes[0]?.id;

    return {
      rootId,
      nodes,
      maxDepth: depth - 1,
      documentId,
    };
  }

  /**
   * Query the context using RLM recursive processing
   */
  async queryContext(
    query: string,
    options: {
      maxDepth?: number;
      maxTokens?: number;
      useCache?: boolean;
      documentIds?: string[];
    } = {}
  ): Promise<RLMAggregatedResult> {
    const startTime = Date.now();
    const { maxDepth = 2, maxTokens = 4000, useCache = true, documentIds } = options;

    // Select trees to query
    const treesToQuery =
      documentIds?.map((id) => this.trees.get(id)).filter(Boolean) ||
      Array.from(this.trees.values());

    if (treesToQuery.length === 0) {
      return {
        originalQuery: query,
        subResults: [],
        finalResponse: "No documents available to query.",
        totalTokensUsed: 0,
        cacheHits: 0,
        processingTime: Date.now() - startTime,
      };
    }

    const context: ProcessingContext = {
      query,
      maxDepth,
      currentDepth: 0,
      useCache,
      accumulatedResults: [],
      tokenBudget: maxTokens,
      tokensUsed: 0,
      cacheHits: 0,
    };

    // Process each tree recursively
    for (const tree of treesToQuery) {
      if (tree) {
        await this.processNode(tree, tree.rootId, context);
      }
    }

    // Aggregate all results into final response
    const finalResponse = await this.aggregateResults(context);

    return {
      originalQuery: query,
      subResults: context.accumulatedResults,
      finalResponse,
      totalTokensUsed: context.tokensUsed,
      cacheHits: context.cacheHits,
      processingTime: Date.now() - startTime,
    };
  }

  /**
   * Recursively process a node in the RLM tree
   */
  private async processNode(
    tree: RLMTree,
    nodeId: string,
    context: ProcessingContext
  ): Promise<RLMQueryResult | null> {
    const node = tree.nodes.get(nodeId);
    if (!node) return null;

    // Check if we've exceeded depth or budget
    if (context.currentDepth > context.maxDepth) {
      return null;
    }

    // First, check if this node is relevant to the query
    const relevance = await this.assessRelevance(node, context.query, context.useCache);
    context.tokensUsed += relevance.tokensUsed;
    if (relevance.cacheHit) context.cacheHits++;

    if (relevance.score < 0.3) {
      // Not relevant enough, skip this branch
      return null;
    }

    // If this is a leaf node or we're at max depth, query directly
    if (node.childIds.length === 0 || context.currentDepth >= context.maxDepth) {
      const result = await this.queryNode(node, context);
      if (result) {
        context.accumulatedResults.push(result);
      }
      return result;
    }

    // Otherwise, recursively process children
    context.currentDepth++;
    const childResults: RLMQueryResult[] = [];

    for (const childId of node.childIds) {
      const childResult = await this.processNode(tree, childId, context);
      if (childResult) {
        childResults.push(childResult);
      }
    }

    context.currentDepth--;

    // If children had results, aggregate them
    if (childResults.length > 0) {
      const aggregated = await this.aggregateNodeResults(node, childResults, context);
      if (aggregated) {
        context.accumulatedResults.push(aggregated);
      }
      return aggregated;
    }

    return null;
  }

  /**
   * Assess relevance of a node to the query
   */
  private async assessRelevance(
    node: RLMNode,
    query: string,
    useCache: boolean
  ): Promise<{ score: number; tokensUsed: number; cacheHit: boolean }> {
    const content = node.summary || node.content.slice(0, 1000);

    const systemPrompt = `You are a relevance assessor. Given a query and content, rate the relevance from 0.0 to 1.0.
Only output a single decimal number between 0.0 and 1.0.`;

    const userPrompt = `Query: ${query}

Content preview:
${content}

Relevance score (0.0-1.0):`;

    try {
      const messages: Anthropic.MessageCreateParamsNonStreaming = {
        model: this.model,
        max_tokens: 10,
        system: this.buildCachedSystem(systemPrompt, useCache),
        messages: [{ role: "user", content: userPrompt }],
      };

      const response = await this.client.messages.create(messages);
      const text =
        response.content[0].type === "text" ? response.content[0].text : "0.5";
      const score = parseFloat(text.trim()) || 0.5;

      const cacheHit =
        (response.usage as { cache_read_input_tokens?: number }).cache_read_input_tokens !== undefined &&
        (response.usage as { cache_read_input_tokens?: number }).cache_read_input_tokens! > 0;

      return {
        score: Math.min(1, Math.max(0, score)),
        tokensUsed: response.usage.input_tokens + response.usage.output_tokens,
        cacheHit,
      };
    } catch (error) {
      console.error("Relevance assessment error:", error);
      return { score: 0.5, tokensUsed: 0, cacheHit: false };
    }
  }

  /**
   * Query a specific node for information
   */
  private async queryNode(
    node: RLMNode,
    context: ProcessingContext
  ): Promise<RLMQueryResult | null> {
    const systemPrompt = `You are a precise information extractor. Answer the query based ONLY on the provided content.
If the content doesn't contain relevant information, say so briefly.
Be concise and factual.`;

    const userPrompt = `Query: ${context.query}

Content:
${node.content}

Answer:`;

    try {
      const messages: Anthropic.MessageCreateParamsNonStreaming = {
        model: this.model,
        max_tokens: Math.min(1000, context.tokenBudget - context.tokensUsed),
        system: this.buildCachedSystem(systemPrompt, context.useCache),
        messages: [{ role: "user", content: userPrompt }],
      };

      const response = await this.client.messages.create(messages);
      const text =
        response.content[0].type === "text" ? response.content[0].text : "";

      context.tokensUsed += response.usage.input_tokens + response.usage.output_tokens;

      const cacheHit =
        (response.usage as { cache_read_input_tokens?: number }).cache_read_input_tokens !== undefined &&
        (response.usage as { cache_read_input_tokens?: number }).cache_read_input_tokens! > 0;
      if (cacheHit) context.cacheHits++;

      return {
        query: context.query,
        nodeId: node.id,
        depth: node.depth,
        response: text,
        relevanceScore: 1.0,
        tokenCount: response.usage.output_tokens,
      };
    } catch (error) {
      console.error("Node query error:", error);
      return null;
    }
  }

  /**
   * Aggregate results from child nodes
   */
  private async aggregateNodeResults(
    parentNode: RLMNode,
    childResults: RLMQueryResult[],
    context: ProcessingContext
  ): Promise<RLMQueryResult | null> {
    const combinedResponses = childResults
      .map((r, i) => `[Source ${i + 1}]: ${r.response}`)
      .join("\n\n");

    const systemPrompt = `You are a synthesis expert. Combine multiple information sources into a coherent response.
Remove redundancy, resolve conflicts, and maintain accuracy.`;

    const userPrompt = `Query: ${context.query}

Information from multiple sources:
${combinedResponses}

Synthesized answer:`;

    try {
      const messages: Anthropic.MessageCreateParamsNonStreaming = {
        model: this.model,
        max_tokens: Math.min(1500, context.tokenBudget - context.tokensUsed),
        system: this.buildCachedSystem(systemPrompt, context.useCache),
        messages: [{ role: "user", content: userPrompt }],
      };

      const response = await this.client.messages.create(messages);
      const text =
        response.content[0].type === "text" ? response.content[0].text : "";

      context.tokensUsed += response.usage.input_tokens + response.usage.output_tokens;

      const cacheHit =
        (response.usage as { cache_read_input_tokens?: number }).cache_read_input_tokens !== undefined &&
        (response.usage as { cache_read_input_tokens?: number }).cache_read_input_tokens! > 0;
      if (cacheHit) context.cacheHits++;

      return {
        query: context.query,
        nodeId: parentNode.id,
        depth: parentNode.depth,
        response: text,
        relevanceScore: Math.max(...childResults.map((r) => r.relevanceScore)),
        tokenCount: response.usage.output_tokens,
      };
    } catch (error) {
      console.error("Aggregation error:", error);
      return null;
    }
  }

  /**
   * Generate final aggregated response from all results
   */
  private async aggregateResults(context: ProcessingContext): Promise<string> {
    if (context.accumulatedResults.length === 0) {
      return "No relevant information found for the query.";
    }

    if (context.accumulatedResults.length === 1) {
      return context.accumulatedResults[0].response;
    }

    // Sort by depth (higher depth = more aggregated) and relevance
    const sortedResults = [...context.accumulatedResults].sort((a, b) => {
      if (b.depth !== a.depth) return b.depth - a.depth;
      return b.relevanceScore - a.relevanceScore;
    });

    // Take top results within token budget
    const topResults = sortedResults.slice(0, 5);
    const combinedResponses = topResults
      .map((r, i) => `[Finding ${i + 1}]: ${r.response}`)
      .join("\n\n");

    const systemPrompt = `You are an expert synthesizer. Create a comprehensive, coherent answer from multiple findings.
Structure the response clearly and cite relevant findings when appropriate.`;

    const userPrompt = `Original Query: ${context.query}

Findings:
${combinedResponses}

Comprehensive Answer:`;

    try {
      const messages: Anthropic.MessageCreateParamsNonStreaming = {
        model: this.model,
        max_tokens: Math.min(2000, context.tokenBudget - context.tokensUsed),
        system: this.buildCachedSystem(systemPrompt, context.useCache),
        messages: [{ role: "user", content: userPrompt }],
      };

      const response = await this.client.messages.create(messages);
      const text =
        response.content[0].type === "text" ? response.content[0].text : "";

      context.tokensUsed += response.usage.input_tokens + response.usage.output_tokens;

      return text;
    } catch (error) {
      console.error("Final aggregation error:", error);
      return topResults[0]?.response || "Error generating response.";
    }
  }

  /**
   * Generate a summary of content
   */
  private async generateSummary(content: string, maxTokens: number): Promise<string> {
    const systemPrompt = `You are a precise summarizer. Create a concise summary that preserves key facts and relationships.
Focus on: main topics, key entities, important facts, and relationships.`;

    const truncatedContent =
      countTokens(content) > 8000 ? content.slice(0, 20000) + "..." : content;

    try {
      const response = await this.client.messages.create({
        model: this.model,
        max_tokens: maxTokens,
        system: systemPrompt,
        messages: [
          {
            role: "user",
            content: `Summarize the following content:\n\n${truncatedContent}`,
          },
        ],
      });

      return response.content[0].type === "text" ? response.content[0].text : "";
    } catch (error) {
      console.error("Summary generation error:", error);
      return content.slice(0, 500) + "...";
    }
  }

  /**
   * Build system prompt with cache control if enabled
   */
  private buildCachedSystem(
    systemPrompt: string,
    useCache: boolean
  ): Anthropic.TextBlockParam[] | string {
    if (!useCache || !this.cacheConfig.enabled) {
      return systemPrompt;
    }

    const tokenCount = countTokens(systemPrompt);
    if (!meetsMinCacheThreshold(tokenCount)) {
      return systemPrompt;
    }

    return [
      {
        type: "text" as const,
        text: systemPrompt,
        cache_control: { type: "ephemeral" as const },
      },
    ];
  }

  /**
   * Get all documents
   */
  getDocuments(): Document[] {
    return Array.from(this.documents.values());
  }

  /**
   * Get a specific document
   */
  getDocument(documentId: string): Document | undefined {
    return this.documents.get(documentId);
  }

  /**
   * Delete a document
   */
  deleteDocument(documentId: string): boolean {
    this.trees.delete(documentId);
    return this.documents.delete(documentId);
  }

  /**
   * Get statistics about the context
   */
  getStats(): {
    documentCount: number;
    totalTokens: number;
    totalChunks: number;
    treeDepths: number[];
  } {
    const documents = this.getDocuments();
    return {
      documentCount: documents.length,
      totalTokens: documents.reduce((sum, doc) => sum + doc.tokenCount, 0),
      totalChunks: documents.reduce((sum, doc) => sum + doc.chunks.length, 0),
      treeDepths: Array.from(this.trees.values()).map((tree) => tree.maxDepth),
    };
  }
}
