/**
 * Intelligent text splitting for RLM processing
 * Splits documents into semantically meaningful chunks
 */

import { countTokens } from "./tokenizer.js";
import type { DocumentChunk, ChunkMetadata } from "../types/index.js";

interface SplitOptions {
  maxTokens: number;
  overlapTokens: number;
  respectBoundaries: boolean;
  contentType: "text" | "markdown" | "code" | "json";
}

/**
 * Split text into chunks based on semantic boundaries
 */
export function splitDocument(
  content: string,
  documentId: string,
  options: SplitOptions
): DocumentChunk[] {
  const { maxTokens, overlapTokens, respectBoundaries, contentType } = options;

  // Choose splitting strategy based on content type
  let rawChunks: string[];

  if (respectBoundaries) {
    switch (contentType) {
      case "markdown":
        rawChunks = splitMarkdown(content, maxTokens, overlapTokens);
        break;
      case "code":
        rawChunks = splitCode(content, maxTokens, overlapTokens);
        break;
      case "json":
        rawChunks = splitJson(content, maxTokens, overlapTokens);
        break;
      default:
        rawChunks = splitText(content, maxTokens, overlapTokens);
    }
  } else {
    rawChunks = splitBySize(content, maxTokens, overlapTokens);
  }

  // Convert to DocumentChunk objects
  const chunks: DocumentChunk[] = [];
  let currentOffset = 0;

  for (let i = 0; i < rawChunks.length; i++) {
    const chunkContent = rawChunks[i];
    const tokenCount = countTokens(chunkContent);
    const startOffset = content.indexOf(chunkContent, currentOffset);
    const endOffset = startOffset + chunkContent.length;

    const metadata: ChunkMetadata = {
      documentId,
      chunkIndex: i,
      totalChunks: rawChunks.length,
      startOffset,
      endOffset,
      depth: 0, // Will be updated during RLM processing
      sectionTitle: extractSectionTitle(chunkContent, contentType),
    };

    chunks.push({
      id: `${documentId}-chunk-${i}`,
      content: chunkContent,
      tokenCount,
      metadata,
    });

    currentOffset = startOffset + 1;
  }

  return chunks;
}

/**
 * Split markdown by headers and sections
 */
function splitMarkdown(content: string, maxTokens: number, overlapTokens: number): string[] {
  const chunks: string[] = [];

  // Split by major headers (##, ###, etc.)
  const headerPattern = /^(#{1,6})\s+(.+)$/gm;
  const sections: string[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = headerPattern.exec(content)) !== null) {
    if (lastIndex < match.index) {
      const section = content.slice(lastIndex, match.index).trim();
      if (section) sections.push(section);
    }
    lastIndex = match.index;
  }

  // Add remaining content
  if (lastIndex < content.length) {
    const section = content.slice(lastIndex).trim();
    if (section) sections.push(section);
  }

  // If no sections found, fall back to paragraph splitting
  if (sections.length === 0) {
    return splitText(content, maxTokens, overlapTokens);
  }

  // Merge small sections and split large ones
  let currentChunk = "";
  for (const section of sections) {
    const sectionTokens = countTokens(section);
    const currentTokens = countTokens(currentChunk);

    if (currentTokens + sectionTokens <= maxTokens) {
      currentChunk += (currentChunk ? "\n\n" : "") + section;
    } else {
      if (currentChunk) {
        chunks.push(currentChunk);
      }

      if (sectionTokens > maxTokens) {
        // Split large section by paragraphs
        const subChunks = splitText(section, maxTokens, overlapTokens);
        chunks.push(...subChunks);
        currentChunk = "";
      } else {
        currentChunk = section;
      }
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks;
}

/**
 * Split code by functions, classes, and logical blocks
 */
function splitCode(content: string, maxTokens: number, overlapTokens: number): string[] {
  const chunks: string[] = [];

  // Common code block patterns
  const patterns = [
    /^(?:export\s+)?(?:async\s+)?function\s+\w+/gm, // Functions
    /^(?:export\s+)?class\s+\w+/gm, // Classes
    /^(?:export\s+)?(?:const|let|var)\s+\w+\s*=/gm, // Variable declarations
    /^\/\*\*[\s\S]*?\*\//gm, // JSDoc comments
  ];

  // Find all potential split points
  const splitPoints: number[] = [0];
  for (const pattern of patterns) {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(content)) !== null) {
      splitPoints.push(match.index);
    }
  }
  splitPoints.push(content.length);

  // Sort and deduplicate
  const uniquePoints = [...new Set(splitPoints)].sort((a, b) => a - b);

  // Create chunks from split points
  let currentChunk = "";
  let lastPoint = 0;

  for (let i = 1; i < uniquePoints.length; i++) {
    const segment = content.slice(lastPoint, uniquePoints[i]);
    const segmentTokens = countTokens(segment);
    const currentTokens = countTokens(currentChunk);

    if (currentTokens + segmentTokens <= maxTokens) {
      currentChunk += segment;
    } else {
      if (currentChunk) {
        chunks.push(currentChunk.trim());
      }

      if (segmentTokens > maxTokens) {
        // Fall back to size-based splitting for very large blocks
        const subChunks = splitBySize(segment, maxTokens, overlapTokens);
        chunks.push(...subChunks);
        currentChunk = "";
      } else {
        currentChunk = segment;
      }
    }

    lastPoint = uniquePoints[i];
  }

  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }

  return chunks.filter((c) => c.length > 0);
}

/**
 * Split JSON by top-level keys or array elements
 */
function splitJson(content: string, maxTokens: number, overlapTokens: number): string[] {
  try {
    const parsed = JSON.parse(content);

    if (Array.isArray(parsed)) {
      // Split array into chunks
      return splitJsonArray(parsed, maxTokens);
    } else if (typeof parsed === "object" && parsed !== null) {
      // Split object by keys
      return splitJsonObject(parsed, maxTokens);
    }
  } catch {
    // If JSON parsing fails, fall back to text splitting
  }

  return splitText(content, maxTokens, overlapTokens);
}

function splitJsonArray(arr: unknown[], maxTokens: number): string[] {
  const chunks: string[] = [];
  let currentBatch: unknown[] = [];

  for (const item of arr) {
    const itemStr = JSON.stringify(item, null, 2);
    const batchStr = JSON.stringify(currentBatch, null, 2);

    if (countTokens(batchStr + itemStr) <= maxTokens) {
      currentBatch.push(item);
    } else {
      if (currentBatch.length > 0) {
        chunks.push(JSON.stringify(currentBatch, null, 2));
      }
      currentBatch = [item];
    }
  }

  if (currentBatch.length > 0) {
    chunks.push(JSON.stringify(currentBatch, null, 2));
  }

  return chunks;
}

function splitJsonObject(obj: Record<string, unknown>, maxTokens: number): string[] {
  const chunks: string[] = [];
  let currentObj: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    const entryStr = JSON.stringify({ [key]: value }, null, 2);
    const currentStr = JSON.stringify(currentObj, null, 2);

    if (countTokens(currentStr + entryStr) <= maxTokens) {
      currentObj[key] = value;
    } else {
      if (Object.keys(currentObj).length > 0) {
        chunks.push(JSON.stringify(currentObj, null, 2));
      }

      if (countTokens(entryStr) > maxTokens) {
        // Value is too large, stringify it alone
        chunks.push(JSON.stringify({ [key]: value }, null, 2));
        currentObj = {};
      } else {
        currentObj = { [key]: value };
      }
    }
  }

  if (Object.keys(currentObj).length > 0) {
    chunks.push(JSON.stringify(currentObj, null, 2));
  }

  return chunks;
}

/**
 * Split plain text by paragraphs and sentences
 */
function splitText(content: string, maxTokens: number, overlapTokens: number): string[] {
  const chunks: string[] = [];

  // Split by double newlines (paragraphs)
  const paragraphs = content.split(/\n\s*\n/);

  let currentChunk = "";

  for (const paragraph of paragraphs) {
    const paraTokens = countTokens(paragraph);
    const currentTokens = countTokens(currentChunk);

    if (currentTokens + paraTokens <= maxTokens) {
      currentChunk += (currentChunk ? "\n\n" : "") + paragraph;
    } else {
      if (currentChunk) {
        chunks.push(currentChunk);
      }

      if (paraTokens > maxTokens) {
        // Split by sentences
        const sentences = splitBySentences(paragraph, maxTokens, overlapTokens);
        chunks.push(...sentences);
        currentChunk = "";
      } else {
        currentChunk = paragraph;
      }
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks;
}

/**
 * Split by sentences
 */
function splitBySentences(text: string, maxTokens: number, overlapTokens: number): string[] {
  const chunks: string[] = [];

  // Simple sentence splitting (handles . ! ?)
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];

  let currentChunk = "";

  for (const sentence of sentences) {
    const sentenceTokens = countTokens(sentence);
    const currentTokens = countTokens(currentChunk);

    if (currentTokens + sentenceTokens <= maxTokens) {
      currentChunk += sentence;
    } else {
      if (currentChunk) {
        chunks.push(currentChunk.trim());
      }

      if (sentenceTokens > maxTokens) {
        // Fall back to size-based splitting
        const subChunks = splitBySize(sentence, maxTokens, overlapTokens);
        chunks.push(...subChunks);
        currentChunk = "";
      } else {
        currentChunk = sentence;
      }
    }
  }

  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}

/**
 * Simple size-based splitting (fallback)
 */
function splitBySize(content: string, maxTokens: number, overlapTokens: number): string[] {
  const chunks: string[] = [];
  const words = content.split(/\s+/);

  let currentChunk = "";

  for (const word of words) {
    const testChunk = currentChunk ? `${currentChunk} ${word}` : word;

    if (countTokens(testChunk) <= maxTokens) {
      currentChunk = testChunk;
    } else {
      if (currentChunk) {
        chunks.push(currentChunk);

        // Add overlap from the end of the current chunk
        if (overlapTokens > 0) {
          const overlapWords = currentChunk.split(/\s+/).slice(-Math.ceil(overlapTokens / 2));
          currentChunk = overlapWords.join(" ") + " " + word;
        } else {
          currentChunk = word;
        }
      } else {
        // Single word exceeds max tokens - include it anyway
        chunks.push(word);
        currentChunk = "";
      }
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks;
}

/**
 * Extract section title from chunk content
 */
function extractSectionTitle(
  content: string,
  contentType: "text" | "markdown" | "code" | "json"
): string | undefined {
  const firstLine = content.split("\n")[0].trim();

  switch (contentType) {
    case "markdown": {
      const headerMatch = firstLine.match(/^#{1,6}\s+(.+)$/);
      if (headerMatch) return headerMatch[1];
      break;
    }
    case "code": {
      const funcMatch = firstLine.match(/(?:function|class|const|let|var)\s+(\w+)/);
      if (funcMatch) return funcMatch[1];
      break;
    }
  }

  // Return first 50 chars as fallback
  return firstLine.length > 50 ? firstLine.slice(0, 50) + "..." : firstLine || undefined;
}
