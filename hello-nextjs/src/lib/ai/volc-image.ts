/**
 * Volcano Engine Image Generation API wrapper.
 * Handles image generation using Seedream model.
 */

import type {
  VolcImageGenerationResponse,
} from "@/types/ai";

// Configuration
const VOLC_ACCESS_KEY = process.env.VOLC_ACCESS_KEY;
const VOLC_SECRET_KEY = process.env.VOLC_SECRET_KEY;
const VOLC_IMAGE_BASE_URL = process.env.VOLC_IMAGE_BASE_URL || "https://visual.volcengineapi.com";
const VOLC_REGION = process.env.VOLC_REGION || "cn-north-1";

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;
const REQUEST_TIMEOUT_MS = 120000; // 2 minutes for image generation

/**
 * Custom error class for Volcano Engine API errors
 */
export class VolcImageApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public errorCode?: number
  ) {
    super(message);
    this.name = "VolcImageApiError";
  }
}

/**
 * Sleep utility for retry delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Check if the API credentials are configured
 */
export function isVolcImageConfigured(): boolean {
  return !!(VOLC_ACCESS_KEY && VOLC_SECRET_KEY);
}

/**
 * Generate HMAC-SHA256 signature for Volcano Engine API
 */
async function generateSignature(
  secretKey: string,
  message: string
): Promise<string> {
  const encoder = new TextEncoder();
  const keyData = encoder.encode(secretKey);
  const messageData = encoder.encode(message);

  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    keyData,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );

  const signature = await crypto.subtle.sign("HMAC", cryptoKey, messageData);
  return btoa(String.fromCharCode(...new Uint8Array(signature)));
}

/**
 * Build authorization header for Volcano Engine API
 */
async function buildAuthHeaders(
  method: string,
  path: string,
  body: string
): Promise<Record<string, string>> {
  if (!VOLC_ACCESS_KEY || !VOLC_SECRET_KEY) {
    throw new VolcImageApiError("Volcano Engine credentials not configured");
  }

  const now = new Date();
  const dateStr = now.toISOString().replace(/[:-]|\.\d{3}/g, "");
  const shortDate = dateStr.slice(0, 8);

  // Build canonical request
  const hashedPayload = await hashSHA256(body);
  const canonicalHeaders = `content-type:application/json\nhost:visual.volcengineapi.com\nx-content-sha256:${hashedPayload}\nx-date:${dateStr}\n`;
  const signedHeaders = "content-type;host;x-content-sha256;x-date";

  const canonicalRequest = `${method}\n${path}\n\n${canonicalHeaders}\n${signedHeaders}\n${hashedPayload}`;

  // Build string to sign
  const credentialScope = `${shortDate}/${VOLC_REGION}/cv/visual/request`;
  const hashedCanonicalRequest = await hashSHA256(canonicalRequest);
  const stringToSign = `HMAC-SHA256\n${dateStr}\n${credentialScope}\n${hashedCanonicalRequest}`;

  // Calculate signature
  const kDate = await generateSignature(VOLC_SECRET_KEY, shortDate);
  const kRegion = await generateSignature(kDate, VOLC_REGION);
  const kService = await generateSignature(kRegion, "cv");
  const kSigning = await generateSignature(kService, "request");
  const signature = await generateSignature(kSigning, stringToSign);

  const authorization = `HMAC-SHA256 Credential=${VOLC_ACCESS_KEY}/${credentialScope}, SignedHeaders=${signedHeaders}, Signature=${signature}`;

  return {
    "Content-Type": "application/json",
    "X-Date": dateStr,
    "X-Content-Sha256": hashedPayload,
    Authorization: authorization,
  };
}

/**
 * Hash a string using SHA-256
 */
async function hashSHA256(message: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(message);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Build style prompt suffix based on video style
 */
function buildStylePrompt(style?: string): string {
  const stylePrompts: Record<string, string> = {
    realistic: ", photorealistic, high quality, natural lighting, sharp details",
    anime: ", anime style, vibrant colors, clean lines, Japanese animation",
    cartoon: ", cartoon style, bright colors, playful, exaggerated features",
    cinematic: ", cinematic, dramatic lighting, film grain, professional cinematography",
    watercolor: ", watercolor painting style, soft edges, delicate colors, artistic",
    oil_painting: ", oil painting style, thick brushstrokes, rich colors, classical art",
    sketch: ", pencil sketch style, detailed linework, grayscale, artistic drawing",
    cyberpunk: ", cyberpunk style, neon lights, futuristic, dark atmosphere, tech aesthetic",
    fantasy: ", fantasy style, magical elements, ethereal, dreamlike, mystical",
    scifi: ", sci-fi style, futuristic, high-tech, space age, advanced technology",
  };

  return stylePrompts[style ?? "realistic"] || stylePrompts.realistic;
}

/**
 * Generate an image from a text description
 * @param prompt - The scene description to generate an image for
 * @param style - Optional visual style
 * @param options - Additional generation options
 * @returns Base64 encoded image data
 */
export async function generateImage(
  prompt: string,
  style?: string,
  options: {
    width?: number;
    height?: number;
    negativePrompt?: string;
    seed?: number;
  } = {}
): Promise<string> {
  if (!isVolcImageConfigured()) {
    throw new VolcImageApiError("Volcano Engine image generation is not configured");
  }

  const stylePrompt = buildStylePrompt(style);
  const fullPrompt = `${prompt}${stylePrompt}`;

  const requestBody = {
    req_key: "high_aes_general_v21_L",
    prompt: fullPrompt,
    negative_prompt: options.negativePrompt ?? "low quality, blurry, distorted, ugly, bad anatomy",
    width: options.width ?? 1024,
    height: options.height ?? 1024,
    use_sr: true,
    sr_scale: 2.0,
    return_url: false,
    ...(options.seed !== undefined && { seed: options.seed }),
  };

  const body = JSON.stringify(requestBody);
  const path = "/";

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const headers = await buildAuthHeaders("POST", path, body);

      const response = await fetch(`${VOLC_IMAGE_BASE_URL}/`, {
        method: "POST",
        headers,
        body,
        signal: controller.signal,
      });

      const data: VolcImageGenerationResponse = await response.json();

      if (data.code !== 10000 && data.code !== 0) {
        throw new VolcImageApiError(
          data.message || `API error: ${data.code}`,
          response.status,
          data.code
        );
      }

      if (!data.data?.binary_data_base64?.[0]) {
        throw new VolcImageApiError("No image data in response");
      }

      clearTimeout(timeoutId);
      return data.data.binary_data_base64[0];
    } catch (error) {
      lastError = error instanceof Error ? error : new Error("Unknown error");

      // Don't retry on auth errors
      if (error instanceof VolcImageApiError && (error.errorCode === 401 || error.errorCode === 403)) {
        throw error;
      }

      // Abort errors shouldn't be retried
      if ((error as Error).name === "AbortError") {
        throw new VolcImageApiError("Request timed out");
      }

      // Retry for other errors
      if (attempt < MAX_RETRIES) {
        console.warn(`Volc Image API attempt ${attempt} failed, retrying...`, error);
        await sleep(RETRY_DELAY_MS * attempt);
      }
    }
  }

  clearTimeout(timeoutId);
  throw new VolcImageApiError(
    `Failed after ${MAX_RETRIES} attempts: ${lastError?.message}`
  );
}

/**
 * Generate an image and return as a Buffer
 * @param prompt - The scene description
 * @param style - Optional visual style
 * @param options - Additional generation options
 * @returns Buffer containing the image data
 */
export async function generateImageBuffer(
  prompt: string,
  style?: string,
  options?: {
    width?: number;
    height?: number;
    negativePrompt?: string;
    seed?: number;
  }
): Promise<Buffer> {
  const base64Data = await generateImage(prompt, style, options);
  return Buffer.from(base64Data, "base64");
}

/**
 * Regenerate an image with different parameters
 * @param prompt - The scene description
 * @param style - Visual style
 * @param seed - Previous seed to modify (or omit for random)
 * @param options - Additional options
 */
export async function regenerateImage(
  prompt: string,
  style?: string,
  seed?: number,
  options?: {
    width?: number;
    height?: number;
    negativePrompt?: string;
  }
): Promise<string> {
  // Use a different seed for regeneration if not specified
  const newSeed = seed ?? Math.floor(Math.random() * 2147483647);

  return generateImage(prompt, style, {
    ...options,
    seed: newSeed,
  });
}
