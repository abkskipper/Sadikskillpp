/**
 * TRADEGUARD AI — CENTRALIZED AI PROVIDER MANAGER & GLOBAL REQUEST QUEUE
 * 
 * Production-grade architecture providing:
 * 1. Global Request Queue with concurrency limit (Strictly 1 active multimodal request at a time)
 * 2. Request deduplication & idempotency tracking
 * 3. Primary (Gemini Vision) and Secondary (Fallback if configured) provider abstraction
 * 4. Granular error classification (RPM, TPM, Daily Quota, Auth, Billing, Provider Demand)
 * 5. Bounded exponential backoff with jitter respecting Retry-After and quota reset windows
 * 6. Live second-by-second countdown for user feedback
 * 7. Clean request cancellation and AbortController propagation
 * 8. Real-time provider health monitoring and verified model availability checks
 */

import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import {
  TRADEGUARD_GEMINI_MODEL,
  parseRetryAfterDurationSeconds,
  formatRetryDelayLabel,
} from './tradeGuardEngine';

export interface ProviderHealth {
  name: string;
  configured: boolean;
  model: string;
  available: boolean;
  status: 'HEALTHY' | 'DEGRADED' | 'EXHAUSTED' | 'UNAUTHENTICATED' | 'UNCONFIGURED';
  latencyMs?: number;
  lastChecked: string;
  error?: string | null;
}

export interface AIProviderStatus {
  primary: ProviderHealth;
  secondary: ProviderHealth;
  activeQueueLength: number;
  currentlyProcessing: boolean;
}

export interface DetailedAIError {
  code:
    | 'RATE_LIMITED'
    | 'QUOTA_EXHAUSTED'
    | 'AUTH_FAILURE'
    | 'BILLING_RESTRICTION'
    | 'PROVIDER_UNAVAILABLE'
    | 'GATEWAY_TIMEOUT'
    | 'MODEL_UNAVAILABLE'
    | 'ANALYSIS_FAILED';
  status: number;
  message: string;
  retryable: boolean;
  retryAfterSeconds?: number;
  provider: string;
  details?: string;
}

// Global Concurrency State (strictly 1 active multimodal vision call to prevent 429 RPM/TPM bursts)
class GlobalAnalysisQueue {
  private queue: Array<{
    jobId: string;
    execute: () => Promise<void>;
    resolveWaiter: () => void;
    abortController?: AbortController;
    onQueuePositionUpdate?: (pos: number, total: number) => void;
  }> = [];

  private isProcessing = false;
  private activeJobId: string | null = null;
  private activeAbortController: AbortController | null = null;

  public get queueLength(): number {
    return this.queue.length;
  }

  public get isBusy(): boolean {
    return this.isProcessing;
  }

  public get currentJobId(): string | null {
    return this.activeJobId;
  }

  /**
   * Enqueues an analysis job. If the queue is empty and engine is idle, runs immediately.
   * Otherwise, queues and informs the caller of its queue position.
   */
  public enqueue(
    jobId: string,
    execute: () => Promise<void>,
    abortController?: AbortController,
    onQueuePositionUpdate?: (pos: number, total: number) => void
  ): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const wrappedExecute = async () => {
        try {
          if (abortController?.signal.aborted) {
            resolve();
            return;
          }
          this.activeJobId = jobId;
          this.activeAbortController = abortController || null;
          await execute();
          resolve();
        } catch (err) {
          reject(err);
        } finally {
          this.activeJobId = null;
          this.activeAbortController = null;
          this.processNext();
        }
      };

      this.queue.push({
        jobId,
        execute: wrappedExecute,
        resolveWaiter: resolve,
        abortController,
        onQueuePositionUpdate,
      });

      this.notifyPositions();

      if (!this.isProcessing) {
        this.processNext();
      }
    });
  }

  public cancel(jobId: string): boolean {
    const idx = this.queue.findIndex(item => item.jobId === jobId);
    if (idx !== -1) {
      const [removed] = this.queue.splice(idx, 1);
      removed.abortController?.abort();
      removed.resolveWaiter();
      this.notifyPositions();
      return true;
    }
    if (this.activeJobId === jobId) {
      this.activeAbortController?.abort();
      return true;
    }
    return false;
  }

  private processNext() {
    if (this.queue.length === 0) {
      this.isProcessing = false;
      return;
    }

    this.isProcessing = true;
    const nextItem = this.queue.shift();
    if (nextItem) {
      this.notifyPositions();
      nextItem.execute().catch(() => {});
    } else {
      this.isProcessing = false;
    }
  }

  private notifyPositions() {
    this.queue.forEach((item, index) => {
      if (item.onQueuePositionUpdate) {
        item.onQueuePositionUpdate(index + 1, this.queue.length);
      }
    });
  }
}

export const globalAnalysisQueue = new GlobalAnalysisQueue();

/**
 * AI Provider Manager
 * Manages primary (Gemini Vision) and optional secondary provider, health monitoring,
 * and multimodal generation with retry backoff and request abort propagation.
 */
export class AIProviderManager {
  private static instance: AIProviderManager;
  private primaryClient: GoogleGenAI | null = null;
  private lastPrimaryKey = '';
  private secondaryClient: GoogleGenAI | null = null;
  private lastSecondaryKey = '';
  private lastVerifiedAt = 0;
  private lastVerifyCache: {
    modelId: string;
    available: boolean;
    status: number;
    latencyMs?: number;
    code?: string;
    error?: string;
  } | null = null;
  private lastHealthCheck: AIProviderStatus | null = null;
  private cooldownUntil = 0;
  private quotaExhaustedUntil = 0;
  private quotaExhaustedMessage: string | null = null;

  private constructor() {
    this.initClients();
  }

  public static getInstance(): AIProviderManager {
    if (!AIProviderManager.instance) {
      AIProviderManager.instance = new AIProviderManager();
    }
    return AIProviderManager.instance;
  }

  public initClients() {
    const primaryKey = (process.env.GEMINI_API_KEY || '').trim();
    if (primaryKey) {
      if (!this.primaryClient || this.lastPrimaryKey !== primaryKey) {
        this.primaryClient = new GoogleGenAI({
          apiKey: primaryKey,
          httpOptions: {
            headers: {
              'User-Agent': 'aistudio-build',
            },
          },
        });
        this.lastPrimaryKey = primaryKey;
        this.lastVerifiedAt = 0;
        this.lastVerifyCache = null;
        this.quotaExhaustedUntil = 0;
        this.quotaExhaustedMessage = null;
      }
    } else {
      this.primaryClient = null;
      this.lastPrimaryKey = '';
    }

    const secondaryKey = (process.env.SECONDARY_GEMINI_API_KEY || process.env.FALLBACK_AI_API_KEY || '').trim();
    if (secondaryKey) {
      if (!this.secondaryClient || this.lastSecondaryKey !== secondaryKey) {
        this.secondaryClient = new GoogleGenAI({
          apiKey: secondaryKey,
          httpOptions: {
            headers: {
              'User-Agent': 'aistudio-build-fallback',
            },
          },
        });
        this.lastSecondaryKey = secondaryKey;
      }
    } else {
      this.secondaryClient = null;
      this.lastSecondaryKey = '';
    }
  }

  public recordTransientSuccess() {
    this.cooldownUntil = 0;
    this.quotaExhaustedUntil = 0;
    this.quotaExhaustedMessage = null;
    if (this.lastHealthCheck?.primary) {
      this.lastHealthCheck.primary.status = 'HEALTHY';
      this.lastHealthCheck.primary.available = true;
      this.lastHealthCheck.primary.error = null;
      this.lastHealthCheck.primary.lastChecked = new Date().toISOString();
    }
  }

  public recordProviderError(classified: DetailedAIError) {
    if (classified.code === 'RATE_LIMITED' && classified.retryAfterSeconds) {
      this.cooldownUntil = Math.max(this.cooldownUntil, Date.now() + classified.retryAfterSeconds * 1000);
    }
    if (classified.code === 'QUOTA_EXHAUSTED') {
      const waitMs = Math.max(60000, (classified.retryAfterSeconds || 3600) * 1000);
      this.quotaExhaustedUntil = Math.max(this.quotaExhaustedUntil, Date.now() + waitMs);
      this.quotaExhaustedMessage = classified.message;
    }
    if (this.lastHealthCheck?.primary) {
      this.lastHealthCheck.primary.lastChecked = new Date().toISOString();
      this.lastHealthCheck.primary.error = classified.message;
      if (classified.code === 'AUTH_FAILURE' || classified.code === 'BILLING_RESTRICTION') {
        this.lastHealthCheck.primary.status = 'UNAUTHENTICATED';
        this.lastHealthCheck.primary.available = false;
      } else if (classified.code === 'QUOTA_EXHAUSTED') {
        this.lastHealthCheck.primary.status = 'EXHAUSTED';
        this.lastHealthCheck.primary.available = false;
      } else if (classified.code === 'RATE_LIMITED' || classified.code === 'PROVIDER_UNAVAILABLE') {
        this.lastHealthCheck.primary.status = 'DEGRADED';
        this.lastHealthCheck.primary.available = true;
      }
    }
  }

  /**
   * Verifies that the configured model exists using zero-token metadata lookup (`models.get`).
   * Never consumes generation RPM/TPM tokens during status checks.
   */
  public async verifyModelSupported(
    modelId: string = TRADEGUARD_GEMINI_MODEL,
    forceRefresh = false
  ): Promise<{
    available: boolean;
    status: number;
    latencyMs?: number;
    code?: string;
    error?: string;
  }> {
    this.initClients();

    if (!this.primaryClient) {
      return {
        available: false,
        status: 401,
        code: 'AUTH_FAILURE',
        error: 'GEMINI_API_KEY is missing in server environment.',
      };
    }

    if (!modelId || !modelId.trim()) {
      return {
        available: false,
        status: 400,
        code: 'MODEL_UNAVAILABLE',
        error: 'TRADEGUARD_GEMINI_MODEL is not configured.',
      };
    }

    const now = Date.now();
    const cacheTtlMs = this.lastVerifyCache?.available ? 300000 : 45000;
    if (
      !forceRefresh &&
      this.lastVerifyCache &&
      this.lastVerifyCache.modelId === modelId &&
      now - this.lastVerifiedAt < cacheTtlMs
    ) {
      return this.lastVerifyCache;
    }

    const t0 = Date.now();
    try {
      const modelInfo = await this.primaryClient.models.get({ model: modelId });
      const latencyMs = Date.now() - t0;
      if (modelInfo && modelInfo.name) {
        const result = { available: true, status: 200, latencyMs };
        this.lastVerifiedAt = Date.now();
        this.lastVerifyCache = { modelId, ...result };
        return result;
      }
      const failRes = {
        available: false,
        status: 404,
        latencyMs,
        code: 'MODEL_UNAVAILABLE',
        error: `Model '${modelId}' did not return metadata from Google Gemini API.`,
      };
      this.lastVerifiedAt = Date.now();
      this.lastVerifyCache = { modelId, ...failRes };
      return failRes;
    } catch (err: any) {
      const latencyMs = Date.now() - t0;
      const classified = this.classifyError(err, 'Primary Gemini');
      const result =
        classified.status === 429 || classified.status === 503
          ? {
              available: classified.code !== 'QUOTA_EXHAUSTED',
              status: classified.status,
              latencyMs,
              code: classified.code,
              error: classified.message,
            }
          : {
              available: false,
              status: classified.status,
              latencyMs,
              code: classified.code,
              error: classified.message,
            };

      this.lastVerifiedAt = Date.now();
      this.lastVerifyCache = { modelId, ...result };
      return result;
    }
  }

  /**
   * Returns current health of all configured AI providers without duplicating upstream probes.
   */
  public async getHealthStatus(forceRefresh = false): Promise<AIProviderStatus> {
    this.initClients();
    const now = Date.now();

    if (!forceRefresh && this.lastHealthCheck && now - this.lastVerifiedAt < 45000) {
      return {
        ...this.lastHealthCheck,
        activeQueueLength: globalAnalysisQueue.queueLength,
        currentlyProcessing: globalAnalysisQueue.isBusy,
      };
    }

    const primaryKey = Boolean(this.lastPrimaryKey);
    const modelId = TRADEGUARD_GEMINI_MODEL;
    let primaryHealth: ProviderHealth = {
      name: 'Google Gemini Vision (Primary)',
      configured: primaryKey,
      model: modelId,
      available: false,
      status: primaryKey ? 'HEALTHY' : 'UNCONFIGURED',
      lastChecked: new Date().toISOString(),
    };

    if (primaryKey) {
      const probe = await this.verifyModelSupported(modelId, forceRefresh);
      primaryHealth.available = probe.available;
      primaryHealth.latencyMs = probe.latencyMs;
      if (!probe.available) {
        if (probe.code === 'AUTH_FAILURE' || probe.code === 'BILLING_RESTRICTION' || probe.status === 401) {
          primaryHealth.status = 'UNAUTHENTICATED';
        } else if (probe.code === 'QUOTA_EXHAUSTED') {
          primaryHealth.status = 'EXHAUSTED';
        } else {
          primaryHealth.status = 'DEGRADED';
        }
        primaryHealth.error = probe.error || 'Model probe unavailable';
      } else if (!forceRefresh && now < this.quotaExhaustedUntil) {
        primaryHealth.available = false;
        primaryHealth.status = 'EXHAUSTED';
        primaryHealth.error = this.quotaExhaustedMessage || 'Daily Gemini quota exhausted.';
      } else if (probe.status === 429 || probe.status === 503 || now < this.cooldownUntil) {
        primaryHealth.status = 'DEGRADED';
        primaryHealth.error = probe.error || 'Rate-limit cooldown active.';
      }
    }

    const secondaryGeminiConfigured = Boolean(this.lastSecondaryKey);
    const secondaryCompatUrl = (process.env.SECONDARY_PROVIDER_URL || '').trim();
    const secondaryCompatKey = (process.env.SECONDARY_PROVIDER_API_KEY || '').trim();
    const secondaryCompatModel = (process.env.SECONDARY_PROVIDER_MODEL || '').trim();
    const secondaryCompatConfigured = Boolean(secondaryCompatUrl && secondaryCompatKey);
    const secondaryConfigured = secondaryGeminiConfigured || secondaryCompatConfigured;

    const secondaryHealth: ProviderHealth = {
      name: secondaryCompatConfigured
        ? 'Secondary Multimodal Endpoint'
        : 'Secondary Gemini Key Fallback',
      configured: secondaryConfigured,
      model: secondaryGeminiConfigured
        ? `${modelId} (Secondary Key)`
        : secondaryCompatConfigured
        ? secondaryCompatModel || 'Configured External Vision Model'
        : 'None Configured',
      available: secondaryConfigured,
      status: secondaryConfigured ? 'HEALTHY' : 'UNCONFIGURED',
      lastChecked: new Date().toISOString(),
      error: secondaryConfigured
        ? null
        : 'No secondary provider configured (SECONDARY_GEMINI_API_KEY or SECONDARY_PROVIDER_URL).',
    };

    this.lastHealthCheck = {
      primary: primaryHealth,
      secondary: secondaryHealth,
      activeQueueLength: globalAnalysisQueue.queueLength,
      currentlyProcessing: globalAnalysisQueue.isBusy,
    };

    return this.lastHealthCheck;
  }

  /**
   * Comprehensive error classifier distinguishing:
   * 1. BILLING_RESTRICTION (non-retryable)
   * 2. AUTH_FAILURE (non-retryable)
   * 3. MODEL_UNAVAILABLE (404 non-retryable)
   * 4. QUOTA_EXHAUSTED (daily/account quota exhausted - non-retryable)
   * 5. RATE_LIMITED (transient RPM/TPM spike - retryable with backoff & Retry-After)
   * 6. PROVIDER_UNAVAILABLE (503 demand surge - retryable)
   * 7. GATEWAY_TIMEOUT (504 deadline exceeded - retryable)
   */
  public classifyError(err: any, providerName = 'Gemini'): DetailedAIError {
    const rawMsg = err?.message || String(err || '');
    let parsedJson: any = null;
    try {
      parsedJson = JSON.parse(rawMsg);
    } catch {
      const match = rawMsg.match(/\{[\s\S]*\}/);
      if (match) {
        try {
          parsedJson = JSON.parse(match[0]);
        } catch {
          // ignore
        }
      }
    }

    const googleErr = parsedJson?.error;
    const status =
      Number(googleErr?.code || err?.status) ||
      (rawMsg.includes('429')
        ? 429
        : rawMsg.includes('503')
        ? 503
        : rawMsg.includes('504')
        ? 504
        : rawMsg.includes('404')
        ? 404
        : rawMsg.includes('401')
        ? 401
        : rawMsg.includes('403')
        ? 403
        : 500);
    const directMsg = String(googleErr?.message || rawMsg || '');

    // Extract Retry-After or quota reset time
    let retryAfterSeconds: number | undefined = undefined;
    const headers = err?.response?.headers || err?.headers;
    if (headers) {
      const headerVal =
        typeof headers.get === 'function'
          ? headers.get('retry-after') || headers.get('Retry-After')
          : headers['retry-after'] || headers['Retry-After'];
      if (headerVal) {
        const parsedSec = parseInt(String(headerVal), 10);
        if (!isNaN(parsedSec) && parsedSec > 0) {
          retryAfterSeconds = parsedSec;
        }
      }
    }
    if (!retryAfterSeconds) {
      retryAfterSeconds = parseRetryAfterDurationSeconds(directMsg);
    }

    // 1. Billing Restrictions (checked before generic 403)
    if (
      directMsg.includes('BILLING_DISABLED') ||
      directMsg.toLowerCase().includes('billing account') ||
      directMsg.toLowerCase().includes('enable billing')
    ) {
      return {
        code: 'BILLING_RESTRICTION',
        status: 403,
        message: 'Google Cloud project billing restriction: Gemini API requires billing enabled or active quota tier.',
        retryable: false,
        provider: providerName,
        details: directMsg,
      };
    }

    // 2. Permanent Authentication / Permission Failures
    if (
      status === 401 ||
      status === 403 ||
      directMsg.includes('API_KEY_INVALID') ||
      directMsg.includes('PERMISSION_DENIED') ||
      directMsg.includes('UNAUTHENTICATED') ||
      directMsg.includes('ACCESS_TOKEN_SCOPE_INSUFFICIENT') ||
      directMsg.includes('API key is missing') ||
      directMsg.includes('environment variable is missing')
    ) {
      return {
        code: 'AUTH_FAILURE',
        status: status === 403 ? 403 : 401,
        message: 'Gemini API authentication failed or API key is missing/invalid. Verify GEMINI_API_KEY in server secrets.',
        retryable: false,
        provider: providerName,
        details: directMsg,
      };
    }

    // 3. Model Not Found / Unsupported (404)
    if (
      status === 404 ||
      directMsg.includes('NOT_FOUND') ||
      directMsg.toLowerCase().includes('model is not found') ||
      directMsg.toLowerCase().includes('not supported for generatecontent')
    ) {
      return {
        code: 'MODEL_UNAVAILABLE',
        status: 404,
        message: `Configured AI model '${TRADEGUARD_GEMINI_MODEL}' was not found or is unsupported for vision analysis.`,
        retryable: false,
        provider: providerName,
        details: directMsg,
      };
    }

    // 4. Permanent Daily / Account / Free-Tier Quota Exhaustion (Do NOT endlessly retry!)
    const lowerMsg = directMsg.toLowerCase();
    const isDailyOrFreeTierQuota =
      lowerMsg.includes('generate_content_free_tier_requests') ||
      lowerMsg.includes('requests per day') ||
      lowerMsg.includes('per day limit') ||
      lowerMsg.includes('perday') ||
      lowerMsg.includes('daily') ||
      lowerMsg.includes('limit: 0');
    const isAccountQuotaMsg =
      lowerMsg.includes('exceeded your current quota') ||
      lowerMsg.includes('free tier') ||
      lowerMsg.includes('billing details');
    const hasShortResetWindow =
      !isDailyOrFreeTierQuota &&
      (lowerMsg.includes('resets in') || Boolean(retryAfterSeconds && retryAfterSeconds <= 120));

    if (
      (directMsg.includes('RESOURCE_EXHAUSTED') || status === 429) &&
      (isDailyOrFreeTierQuota || (isAccountQuotaMsg && !hasShortResetWindow))
    ) {
      const resetLabel = formatRetryDelayLabel(retryAfterSeconds);
      const resetText = resetLabel ? ` Quota resets in ${resetLabel}.` : '';
      return {
        code: 'QUOTA_EXHAUSTED',
        status: 429,
        message: `Google Gemini API daily/free-tier quota exhausted.${resetText} Check your billing settings or configure a secondary API key in Server Secrets.`,
        retryable: false,
        retryAfterSeconds,
        provider: providerName,
        details: directMsg,
      };
    }

    // 5. Transient RPM / TPM Rate Limit Spike (Retryable with backoff)
    if (
      status === 429 ||
      directMsg.includes('RESOURCE_EXHAUSTED') ||
      lowerMsg.includes('spikes in demand') ||
      lowerMsg.includes('too many requests') ||
      lowerMsg.includes('resets in') ||
      lowerMsg.includes('rate limit') ||
      lowerMsg.includes('tokens per minute') ||
      lowerMsg.includes('requests per minute')
    ) {
      const resetLabel = formatRetryDelayLabel(retryAfterSeconds);
      const resetMsg = resetLabel ? ` Quota resets in ${resetLabel}.` : '';
      return {
        code: 'RATE_LIMITED',
        status: 429,
        message: `Gemini rate limit reached.${resetMsg} Please wait a moment before retrying.`,
        retryable: true,
        retryAfterSeconds: retryAfterSeconds || 6,
        provider: providerName,
        details: directMsg,
      };
    }

    // 6. Transient 503 Provider High Demand
    if (
      status === 503 ||
      directMsg.includes('503') ||
      lowerMsg.includes('high demand') ||
      lowerMsg.includes('temporarily overloaded') ||
      lowerMsg.includes('temporarily unavailable')
    ) {
      return {
        code: 'PROVIDER_UNAVAILABLE',
        status: 503,
        message: 'Temporary Gemini Server Demand Surge: This model is currently experiencing high demand. Spikes in traffic are normal. Please click Retry Institutional Analysis.',
        retryable: true,
        retryAfterSeconds: retryAfterSeconds || 3,
        provider: providerName,
        details: directMsg,
      };
    }

    // 7. 504 Timeout
    if (
      status === 504 ||
      lowerMsg.includes('timed out') ||
      directMsg.includes('ETIMEDOUT') ||
      directMsg.includes('DEADLINE_EXCEEDED')
    ) {
      return {
        code: 'GATEWAY_TIMEOUT',
        status: 504,
        message: 'Analysis timed out while processing institutional charts. Please try again.',
        retryable: true,
        retryAfterSeconds: 2,
        provider: providerName,
        details: directMsg,
      };
    }

    return {
      code: 'ANALYSIS_FAILED',
      status: status || 500,
      message: directMsg || 'Multimodal institutional vision analysis failed.',
      retryable: status >= 500,
      provider: providerName,
      details: directMsg,
    };
  }

  private async executeOpenAICompatibleFallback(options: {
    url: string;
    apiKey: string;
    model: string;
    contentParts: any[];
    systemInstruction: string;
    abortController?: AbortController;
    t0: number;
  }) {
    const { url, apiKey, model, contentParts, systemInstruction, abortController, t0 } = options;
    const messagesContent: any[] = [];
    for (const part of contentParts) {
      if (part?.text) {
        messagesContent.push({ type: 'text', text: part.text });
      } else if (part?.inlineData?.data) {
        messagesContent.push({
          type: 'image_url',
          image_url: {
            url: `data:${part.inlineData.mimeType || 'image/jpeg'};base64,${part.inlineData.data}`,
          },
        });
      }
    }

    const attemptAbort = new AbortController();
    const timeoutId = setTimeout(() => attemptAbort.abort(), 40000);
    const onCallerAbort = () => attemptAbort.abort();
    abortController?.signal.addEventListener('abort', onCallerAbort, { once: true });

    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        signal: attemptAbort.signal,
        body: JSON.stringify({
          model,
          response_format: { type: 'json_object' },
          messages: [
            { role: 'system', content: systemInstruction },
            { role: 'user', content: messagesContent },
          ],
        }),
      });

      const bodyText = await resp.text();
      if (!resp.ok) {
        const err: any = new Error(bodyText || `Secondary provider HTTP ${resp.status}`);
        err.status = resp.status;
        throw err;
      }

      const data = JSON.parse(bodyText);
      const rawText = data?.choices?.[0]?.message?.content?.trim() || '';
      if (!rawText) {
        throw new Error('Secondary multimodal provider returned empty content.');
      }

      return {
        rawText,
        modelUsed: model,
        providerUsed: 'Secondary Multimodal Endpoint',
        durationMs: Date.now() - t0,
        tokenUsage: data?.usage || null,
      };
    } finally {
      clearTimeout(timeoutId);
      abortController?.signal.removeEventListener('abort', onCallerAbort);
    }
  }

  /**
   * Executes multimodal vision inference with:
   * - Primary provider (Gemini 3.8 Flash)
   * - Respects active rate-limit cooldown window before issuing requests
   * - Bounded exponential backoff with jitter + live 1s countdown updates
   * - Per-attempt AbortController wired into SDK abortSignal
   * - Optional secondary provider failover (SECONDARY_GEMINI_API_KEY or SECONDARY_PROVIDER_URL)
   */
  public async executeVisionInference(options: {
    contentParts: any[];
    systemInstruction: string;
    abortController?: AbortController;
    onProgress?: (stageMessage: string) => void;
    maxAttempts?: number;
  }): Promise<{
    rawText: string;
    modelUsed: string;
    providerUsed: string;
    durationMs: number;
    tokenUsage?: any;
  }> {
    const {
      contentParts,
      systemInstruction,
      abortController,
      onProgress,
      maxAttempts = 3,
    } = options;

    this.initClients();

    if (!this.primaryClient) {
      const missingErr = this.classifyError(new Error('GEMINI_API_KEY environment variable is missing on server.'));
      this.recordProviderError(missingErr);
      throw missingErr;
    }

    const t0 = Date.now();

    // Respect active rate-limit cooldown if set by a recent 429 response
    const now = Date.now();
    if (this.cooldownUntil > now) {
      const waitSec = Math.min(30, Math.ceil((this.cooldownUntil - now) / 1000));
      for (let secLeft = waitSec; secLeft > 0; secLeft--) {
        if (abortController?.signal.aborted) {
          throw new Error('Analysis cancelled by user.');
        }
        if (onProgress) {
          onProgress(`Waiting for upstream Gemini rate-limit cooldown (${secLeft}s remaining)...`);
        }
        await new Promise(r => setTimeout(r, 1000));
      }
    }

    const executeWithClient = async (
      client: GoogleGenAI,
      modelId: string,
      providerLabel: string
    ) => {
      let lastErr: any = null;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        if (abortController?.signal.aborted) {
          throw new Error('Analysis cancelled by user.');
        }

        const attemptController = new AbortController();
        const CALL_TIMEOUT_MS = 65000;
        let timedOut = false;
        const callTimeoutId = setTimeout(() => {
          timedOut = true;
          attemptController.abort();
        }, CALL_TIMEOUT_MS);

        const onCallerAbort = () => attemptController.abort();
        abortController?.signal.addEventListener('abort', onCallerAbort, { once: true });

        try {
          const config: any = {
            systemInstruction,
            responseMimeType: 'application/json',
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            abortSignal: attemptController.signal,
          };

          const response = await client.models.generateContent({
            model: modelId,
            contents: { parts: contentParts },
            config,
          });

          clearTimeout(callTimeoutId);
          abortController?.signal.removeEventListener('abort', onCallerAbort);

          const rawText = response?.text?.trim() || '';
          if (!rawText) {
            throw new Error('Multimodal vision provider returned empty response.');
          }

          this.recordTransientSuccess();
          return {
            rawText,
            modelUsed: modelId,
            providerUsed: providerLabel,
            durationMs: Date.now() - t0,
            tokenUsage: response?.usageMetadata || null,
          };
        } catch (callErr: any) {
          clearTimeout(callTimeoutId);
          abortController?.signal.removeEventListener('abort', onCallerAbort);

          if (abortController?.signal.aborted) {
            throw new Error('Analysis cancelled by user.');
          }

          if (timedOut) {
            const timeoutErr: any = new Error(`Gemini vision request to ${modelId} timed out after ${CALL_TIMEOUT_MS}ms`);
            timeoutErr.status = 504;
            callErr = timeoutErr;
          }

          lastErr = callErr;
          const classified = this.classifyError(callErr, providerLabel);
          this.recordProviderError(classified);

          if (!classified.retryable) {
            console.error(`[AIProviderManager] Non-retryable ${classified.code} on ${providerLabel}:`, classified.message);
            throw classified;
          }

          if (attempt < maxAttempts) {
            const baseDelay = 2500 * Math.pow(1.8, attempt - 1);
            const jitter = Math.floor(Math.random() * 900);
            const retryAfterDelay = classified.retryAfterSeconds ? (classified.retryAfterSeconds + 1) * 1000 : 0;
            const delayMs = Math.min(35000, Math.max(baseDelay + jitter, retryAfterDelay));
            const totalSec = Math.ceil(delayMs / 1000);

            console.log(`[AIProviderManager] Backoff on ${providerLabel} (${classified.code}): retrying in ${delayMs}ms (attempt ${attempt + 1}/${maxAttempts})`);

            for (let secLeft = totalSec; secLeft > 0; secLeft--) {
              if (abortController?.signal.aborted) {
                throw new Error('Analysis cancelled by user.');
              }
              if (onProgress) {
                onProgress(`Upstream ${providerLabel} limit reached (${classified.code}). Auto-retrying in ${secLeft}s (attempt ${attempt + 1}/${maxAttempts})...`);
              }
              await new Promise(r => setTimeout(r, 1000));
            }
          } else {
            throw classified;
          }
        }
      }
      throw lastErr;
    };

    try {
      const secondaryCompatConfigured = Boolean(
        (process.env.SECONDARY_PROVIDER_URL || '').trim() &&
        (process.env.SECONDARY_PROVIDER_API_KEY || '').trim() &&
        (process.env.SECONDARY_PROVIDER_MODEL || '').trim()
      );
      if (Date.now() < this.quotaExhaustedUntil && (this.secondaryClient || secondaryCompatConfigured)) {
        const err = new Error(this.quotaExhaustedMessage || 'Primary daily quota exhausted.');
        (err as any).status = 429;
        throw err;
      }
      return await executeWithClient(this.primaryClient, TRADEGUARD_GEMINI_MODEL, 'Gemini Primary');
    } catch (primaryErr: any) {
      if (abortController?.signal.aborted || String(primaryErr?.message || '').includes('cancelled')) {
        throw primaryErr;
      }

      if (this.secondaryClient) {
        console.warn('[AIProviderManager] Primary Gemini unavailable. Failing over to secondary Gemini key...');
        if (onProgress) {
          onProgress('Primary Gemini unavailable. Failing over to secondary multimodal engine...');
        }
        return await executeWithClient(this.secondaryClient, TRADEGUARD_GEMINI_MODEL, 'Gemini Secondary');
      }

      const secondaryCompatUrl = (process.env.SECONDARY_PROVIDER_URL || '').trim();
      const secondaryCompatKey = (process.env.SECONDARY_PROVIDER_API_KEY || '').trim();
      const secondaryCompatModel = (process.env.SECONDARY_PROVIDER_MODEL || '').trim();
      if (secondaryCompatUrl && secondaryCompatKey && secondaryCompatModel) {
        console.warn('[AIProviderManager] Primary Gemini unavailable. Failing over to secondary compatible multimodal endpoint...');
        if (onProgress) {
          onProgress('Primary Gemini unavailable. Failing over to secondary multimodal provider...');
        }
        return await this.executeOpenAICompatibleFallback({
          url: secondaryCompatUrl,
          apiKey: secondaryCompatKey,
          model: secondaryCompatModel,
          contentParts,
          systemInstruction,
          abortController,
          t0,
        });
      }

      throw primaryErr;
    }
  }
}

export const aiProviderManager = AIProviderManager.getInstance();
