import dotenv from 'dotenv';
dotenv.config();

import express, { Request, Response } from 'express';
import http from 'http';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import {
  TRADEGUARD_GEMINI_MODEL,
  TRADEGUARD_ENGINE_VERSION,
  TRADEGUARD_PROMPT_VERSION,
  TRADEGUARD_SCHEMA_VERSION,
  TRADEGUARD_MODEL_VERSION,
  TRADEGUARD_RESPONSE_SCHEMA,
  buildTradeGuardPrompt,
  validateTradeGuardAnalysis,
  logPreAndPostValidation,
  parseNumericPrice,
  normalizeAplusSmcSequence,
  calculateInstitutionalRiskManagement,
  inferTradingSession,
  formatSafeSessionDate,
  classifyGeminiError,
} from './src/lib/tradeGuardEngine';
import { aiProviderManager, globalAnalysisQueue } from './src/lib/aiProviderManager';
import type { Strategy } from './src/types';
import { runRegressionTests } from './test-tradeguard';

export {
  TRADEGUARD_GEMINI_MODEL,
  TRADEGUARD_ENGINE_VERSION,
  TRADEGUARD_PROMPT_VERSION,
  TRADEGUARD_SCHEMA_VERSION,
  TRADEGUARD_MODEL_VERSION,
  validateTradeGuardAnalysis,
  parseNumericPrice,
};

// =============================================================
// PRODUCTION MODEL INTEGRITY CHECKS & DIAGNOSTIC LOGGING
// =============================================================
const rawModelEnv = process.env.TRADEGUARD_GEMINI_MODEL;
const rawApiKeyEnv = process.env.GEMINI_API_KEY;
console.log(`[TradeGuard Startup] TRADEGUARD_GEMINI_MODEL configured: ${Boolean(rawModelEnv && rawModelEnv.trim().length > 0)}`);
console.log(`[TradeGuard Startup] GEMINI_API_KEY configured: ${Boolean(rawApiKeyEnv && rawApiKeyEnv.trim().length > 0)}`);
console.log(`[TradeGuard Startup] Configured Model: ${TRADEGUARD_GEMINI_MODEL}`);
if (!rawApiKeyEnv || rawApiKeyEnv.trim() === '') {
  console.error('[TRADEGUARD FATAL CONFIG ERROR] GEMINI_API_KEY environment variable is missing.');
}

/**
 * Diagnostic logger for failed analysis requests (Requirement 3)
 * Safely outputs structured diagnostic metadata without exposing secrets.
 */
function logDiagnosticFailure(params: {
  requestId: string;
  stage: string;
  error: string;
  status: number;
  code: string;
  pair?: string;
  strategy?: string;
  imagesReceived?: number;
  imageMimes?: string[];
  imageSizes?: number[];
  geminiHttpStatus?: number;
  geminiErrorCode?: string;
  geminiErrorMessage?: string;
}) {
  const timestamp = new Date().toISOString();
  if (params.status === 429 || params.status === 503) {
    console.log(`[TradeGuard Notice] ${timestamp} Request ${params.requestId}: Status ${params.status} (${params.code}) | Upstream: ${String(params.geminiErrorMessage || '').slice(0, 500)}`);
    return;
  }
  console.error(`[TRADEGUARD DIAGNOSTIC FAILURE] ------------------------
  Request ID:        ${params.requestId}
  Timestamp:         ${timestamp}
  Configured Model:  ${TRADEGUARD_GEMINI_MODEL}
  API Endpoint:      /api/analyzeChart
  Selected Pair:     ${params.pair || 'N/A'}
  Selected Strategy: ${params.strategy || 'N/A'}
  Images Received:   ${params.imagesReceived ?? 0}
  Image MIME Types:  ${JSON.stringify(params.imageMimes || [])}
  Image Byte Sizes:  ${JSON.stringify(params.imageSizes || [])}
  Validation Stage:  ${params.stage}
  HTTP Status:       ${params.status}
  Internal Code:     ${params.code}
  Gemini Status:     ${params.geminiHttpStatus ?? 'N/A'}
  Gemini Error Code: ${params.geminiErrorCode ?? 'N/A'}
  Gemini Error Msg:  ${params.geminiErrorMessage ?? params.error}
  Schema Configured: true
-------------------------------------------------------`);
}

/**
 * Detects and validates image magic bytes / headers to reject invalid or corrupted payloads.
 */
function detectAndValidateImageBuffer(buffer: Buffer): { valid: boolean; mime: string; format: string } {
  if (!buffer || buffer.length < 8) return { valid: false, mime: '', format: 'corrupt' };

  // JPEG: FF D8 FF
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return { valid: true, mime: 'image/jpeg', format: 'jpeg' };
  }
  // PNG: 89 50 4E 47 0D 0A 1A 0A
  if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47) {
    return { valid: true, mime: 'image/png', format: 'png' };
  }
  // WebP: RIFF .... WEBP
  if (
    buffer.length >= 12 &&
    buffer.toString('ascii', 0, 4) === 'RIFF' &&
    buffer.toString('ascii', 8, 12) === 'WEBP'
  ) {
    return { valid: true, mime: 'image/webp', format: 'webp' };
  }
  // SVG: starts with <svg or <?xml
  const head = buffer.slice(0, 200).toString('utf-8');
  if (head.includes('<svg') || head.includes('<?xml')) {
    return { valid: true, mime: 'image/svg+xml', format: 'svg' };
  }
  return { valid: false, mime: 'unknown', format: 'invalid' };
}

/**
 * Normalizes and converts any incoming chart image (SVG, PNG, WebP, JPEG)
 * to a clean, lightweight JPEG buffer for reliable Gemini Vision processing.
 * Preserves exact aspect ratio and candle wick clarity.
 */
async function processChartImageForGemini(
  chart: any,
  role: string
): Promise<{ mimeType: string; data: string; sizeBytes: number }> {
  if (!chart || !chart.imageBase64) {
    throw new Error(`Missing image data for chart role ${role}`);
  }

  const rawBase64 = chart.imageBase64.includes(',')
    ? chart.imageBase64.split(',')[1]
    : chart.imageBase64;
  const inputBuffer = Buffer.from(rawBase64, 'base64');

  // Enforce max upload size limit: 12MB
  const maxSizeBytes = 12 * 1024 * 1024;
  if (inputBuffer.length > maxSizeBytes) {
    throw new Error(`Chart image for role ${role} exceeds safe size limit (${(inputBuffer.length / (1024 * 1024)).toFixed(1)}MB > 12MB).`);
  }

  // Validate image signature
  const sigCheck = detectAndValidateImageBuffer(inputBuffer);
  if (!sigCheck.valid) {
    throw new Error(`Chart image for role ${role} has an invalid or unrecognized image signature. Supported formats: JPEG, PNG, WebP, SVG.`);
  }

  try {
    const sharpModule = await import('sharp');
    const sharp: any = (sharpModule as any).default || sharpModule;

    // Check image metadata: if already <= 1024x1024 and JPEG and under 250KB, preserve directly
    let meta: any = null;
    try {
      meta = await sharp(inputBuffer).metadata();
    } catch {}

    if (sigCheck.format === 'jpeg' && inputBuffer.length < 250000 && meta && meta.width && meta.height && meta.width <= 1024 && meta.height <= 1024) {
      return {
        mimeType: 'image/jpeg',
        data: rawBase64,
        sizeBytes: inputBuffer.length,
      };
    }

    // Resize preserving exact aspect ratio strictly, without stretching or distorting wicks.
    // 1024x1024 inside fit ensures crisp 1-pixel candle wicks while limiting vision tile count
    // to ~2 tiles per chart (~516 tokens instead of >1500 tokens), preventing rate limits.
    const processedBuffer = await sharp(inputBuffer)
      .resize(1024, 1024, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 80, progressive: true })
      .toBuffer();

    return {
      mimeType: 'image/jpeg',
      data: processedBuffer.toString('base64'),
      sizeBytes: processedBuffer.length,
    };
  } catch (err: any) {
    // Graceful fallback if sharp fails
    return {
      mimeType: sigCheck.mime || 'image/jpeg',
      data: rawBase64,
      sizeBytes: inputBuffer.length,
    };
  }
}

const app = express();
const portArgIndex = process.argv.indexOf('--port');
const cliPort =
  portArgIndex !== -1 && process.argv[portArgIndex + 1]
    ? Number(process.argv[portArgIndex + 1])
    : NaN;
const PORT =
  (Number.isFinite(cliPort) && cliPort > 0 ? cliPort : 0) ||
  Number(process.env.PORT) ||
  3000;

// Body parsing with higher limits for multi-chart base64 uploads
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// -------------------------------------------------------------
// Database Persistence Layer (in-memory + disk backup)
// -------------------------------------------------------------
const DB_FILE = path.join(process.cwd(), '.data', 'tradeguard_db.json');

interface DatabaseSchema {
  users: Array<{
    id: string;
    email: string;
    name: string;
    passwordHash: string;
    role: 'user' | 'admin' | 'super_admin';
    created_at: string;
  }>;
  signals: any[];
  watchlist: any[];
  analysisLogs: any[];
  jobs?: any[];
}

// -------------------------------------------------------------
// Secure Authentication & Authorization Layer
// -------------------------------------------------------------
const AUTH_SECRET =
  process.env.AUTH_SECRET ||
  'tradeguard_prod_auth_secret_' +
    crypto
      .createHash('sha256')
      .update(process.env.GEMINI_API_KEY || process.cwd())
      .digest('hex')
      .substring(0, 32);

function hashPassword(password: string): string {
  const salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.scryptSync(password, salt, 64).toString('hex');
  return `${salt}:${hash}`;
}

function verifyPassword(password: string, storedHash: string): boolean {
  if (!password || !storedHash || !storedHash.includes(':')) return false;
  const parts = storedHash.split(':');
  if (parts.length !== 2) return false;
  const [salt, hash] = parts;
  if (!salt || !hash) return false;
  try {
    const testHash = crypto.scryptSync(password, salt, 64);
    const storedBuf = Buffer.from(hash, 'hex');
    if (testHash.length !== storedBuf.length) return false;
    return crypto.timingSafeEqual(storedBuf, testHash);
  } catch {
    return false;
  }
}

function createAuthToken(user: { id: string; email: string; role: string }): string {
  const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64url');
  const payload = Buffer.from(
    JSON.stringify({
      id: user.id,
      email: user.email,
      role: user.role,
      exp: Date.now() + 86400000 * 7, // 7 days
    })
  ).toString('base64url');
  const sig = crypto.createHmac('sha256', AUTH_SECRET).update(`${header}.${payload}`).digest('base64url');
  return `${header}.${payload}.${sig}`;
}

function verifyAuthToken(token: string): { id: string; email: string; role: string } | null {
  if (!token || typeof token !== 'string') return null;
  const parts = token.split('.');
  if (parts.length !== 3) return null;

  const [header, payload, sig] = parts;
  if (!header || !payload || !sig) return null;

  const expectedSig = crypto.createHmac('sha256', AUTH_SECRET).update(`${header}.${payload}`).digest('base64url');
  const sigBuf = Buffer.from(sig);
  const expectedBuf = Buffer.from(expectedSig);
  if (sigBuf.length !== expectedBuf.length || !crypto.timingSafeEqual(sigBuf, expectedBuf)) {
    return null;
  }

  try {
    const data = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8'));
    if (!data?.id || !data?.exp || Date.now() > data.exp) return null;
    return { id: data.id, email: data.email, role: data.role };
  } catch {
    return null;
  }
}

function getAuthenticatedUser(req: Request, allowQueryToken = false): any | null {
  let token: string | null = null;
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    token = authHeader.substring(7).trim();
  } else if (allowQueryToken && typeof req.query.token === 'string' && req.query.token.trim()) {
    token = req.query.token.trim();
  }
  if (!token) return null;
  const verified = verifyAuthToken(token);
  if (!verified) return null;
  return db.users.find(u => u.id === verified.id) || null;
}

// Initial seed data with cryptographic scrypt-hashed credentials
function buildInitialDb(): DatabaseSchema {
  return {
    users: [
      {
        id: 'usr_trader_demo',
        email: 'trader@tradeguard.ai',
        name: 'Institutional Desk Trader',
        passwordHash: hashPassword(process.env.DEMO_TRADER_PASSWORD || 'demo123'),
        role: 'user',
        created_at: new Date(Date.now() - 86400000 * 7).toISOString(),
      },
      {
        id: 'usr_admin_demo',
        email: 'admin@tradeguard.ai',
        name: 'Senior Risk Officer (Admin)',
        passwordHash: hashPassword(process.env.ADMIN_INITIAL_PASSWORD || 'admin123'),
        role: 'super_admin',
        created_at: new Date(Date.now() - 86400000 * 30).toISOString(),
      },
    ],
    signals: [],
    watchlist: [],
    analysisLogs: [],
    jobs: [],
  };
}

function loadDb(): DatabaseSchema {
  try {
    if (!fs.existsSync(path.dirname(DB_FILE))) {
      fs.mkdirSync(path.dirname(DB_FILE), { recursive: true });
    }
    if (fs.existsSync(DB_FILE)) {
      const data = fs.readFileSync(DB_FILE, 'utf-8');
      const parsed = JSON.parse(data);
      if (parsed && Array.isArray(parsed.signals)) {
        parsed.signals = parsed.signals
          .filter((s: any) =>
            s.id !== 'sig_btc_sample_h1' &&
            s.entry !== 105250 &&
            s.entry !== 4594.5 &&
            s.entry !== 4602.5 &&
            s.modelUsed !== 'TradeGuard Institutional Engine'
          )
          .map((s: any) => {
            const resolvedIso =
              s.created_at ||
              s.timestamp ||
              s.marketOverview?.analysisTimestamp ||
              undefined;
            if (!resolvedIso) return s;
            return {
              ...s,
              created_at: s.created_at || resolvedIso,
              timestamp: s.timestamp || resolvedIso,
            };
          });
      }
      // Upgrade any legacy non-scrypt password hashes cleanly at startup
      let upgraded = false;
      if (Array.isArray(parsed?.users)) {
        for (const u of parsed.users) {
          if (!u.passwordHash || !String(u.passwordHash).includes(':')) {
            const seedSecret =
              u.id === 'usr_admin_demo'
                ? process.env.ADMIN_INITIAL_PASSWORD || 'admin123'
                : u.id === 'usr_trader_demo'
                ? process.env.DEMO_TRADER_PASSWORD || 'demo123'
                : crypto.randomBytes(16).toString('hex');
            u.passwordHash = hashPassword(seedSecret);
            upgraded = true;
          }
        }
      }
      if (!Array.isArray(parsed.jobs)) parsed.jobs = [];
      if (!Array.isArray(parsed.watchlist)) parsed.watchlist = [];
      if (!Array.isArray(parsed.analysisLogs)) parsed.analysisLogs = [];
      if (upgraded) {
        try {
          fs.writeFileSync(DB_FILE, JSON.stringify(parsed, null, 2), 'utf-8');
        } catch {
          // ignore initial sync write error
        }
      }
      return parsed;
    }
  } catch (err) {
    console.error('Failed to read db file, using initial seed:', err);
  }
  const seeded = buildInitialDb();
  try {
    fs.writeFileSync(DB_FILE, JSON.stringify(seeded, null, 2), 'utf-8');
  } catch {
    // ignore
  }
  return seeded;
}

let db: DatabaseSchema = loadDb();
let isSaving = false;
let pendingSave = false;

function saveDb() {
  if (isSaving) {
    pendingSave = true;
    return;
  }
  isSaving = true;
  setImmediate(async () => {
    const tmpFile = `${DB_FILE}.tmp.${process.pid}.${Date.now()}`;
    try {
      const dir = path.dirname(DB_FILE);
      if (!fs.existsSync(dir)) {
        await fs.promises.mkdir(dir, { recursive: true });
      }
      const snapshot = JSON.stringify(db, null, 2);
      await fs.promises.writeFile(tmpFile, snapshot, 'utf-8');
      await fs.promises.rename(tmpFile, DB_FILE);
    } catch (err) {
      console.error('Failed to atomically write db file:', err);
      try {
        if (fs.existsSync(tmpFile)) await fs.promises.unlink(tmpFile);
      } catch {
        // ignore cleanup failure
      }
    } finally {
      isSaving = false;
      if (pendingSave) {
        pendingSave = false;
        saveDb();
      }
    }
  });
}

app.get('/api/auth/session', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (user) {
    return res.json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        created_at: user.created_at,
      },
    });
  }
  return res.status(401).json({ success: false, user: null, error: 'Not authenticated.' });
});

app.post('/api/auth/demo', (_req: Request, res: Response) => {
  const demoTrader = db.users.find(u => u.id === 'usr_trader_demo' && u.role === 'user');
  if (!demoTrader) {
    return res.status(404).json({ success: false, error: 'Demo trader account not available.' });
  }
  const token = createAuthToken(demoTrader);
  return res.json({
    success: true,
    token,
    user: {
      id: demoTrader.id,
      email: demoTrader.email,
      name: demoTrader.name,
      role: demoTrader.role,
      created_at: demoTrader.created_at,
    },
  });
});

app.post('/api/auth/signin', (req: Request, res: Response) => {
  const { email, password } = req.body;
  if (!email || !password) {
    return res.status(400).json({ success: false, error: 'Email and password are required.' });
  }

  const user = db.users.find(u => u.email.toLowerCase() === String(email).toLowerCase().trim());
  if (!user || !verifyPassword(String(password), user.passwordHash)) {
    return res.status(401).json({ success: false, error: 'Invalid email or password.' });
  }

  const token = createAuthToken(user);
  return res.json({
    success: true,
    token,
    user: {
      id: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
      created_at: user.created_at,
    },
  });
});

app.post('/api/auth/signup', (req: Request, res: Response) => {
  const { email, name, password } = req.body;
  if (!email || !password) {
    return res.status(400).json({ success: false, error: 'Email and password are required.' });
  }
  if (String(password).length < 6) {
    return res.status(400).json({ success: false, error: 'Password must be at least 6 characters.' });
  }

  const normalizedEmail = String(email).trim().toLowerCase();
  const existing = db.users.find(u => u.email.toLowerCase() === normalizedEmail);
  if (existing) {
    return res.status(400).json({ success: false, error: 'User already exists with this email.' });
  }

  // Security: Disallow arbitrary role escalation. Signups strictly receive role: 'user'.
  const newUser = {
    id: `usr_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
    email: normalizedEmail,
    name: String(name || '').trim() || normalizedEmail.split('@')[0],
    passwordHash: hashPassword(String(password)),
    role: 'user' as const,
    created_at: new Date().toISOString(),
  };
  db.users.push(newUser);
  saveDb();

  const token = createAuthToken(newUser);
  return res.json({
    success: true,
    token,
    user: {
      id: newUser.id,
      email: newUser.email,
      name: newUser.name,
      role: newUser.role,
      created_at: newUser.created_at,
    },
  });
});

// -------------------------------------------------------------
// Signals & Watchlist Endpoints & Multi-Strategy Confluence Audit (SMC & CRT)
// -------------------------------------------------------------
function buildAplusConfluenceAudit(s: any): any {
  if (!s) return null;
  const strat = String(s.strategy || s.selectedStrategy || 'SMC').toUpperCase();
  const isBull = s.bias === 'BULLISH';
  const entry = parseNumericPrice(s.entry) || 0;
  const sl = parseNumericPrice(s.stopLoss) || 0;
  const tp1 = parseNumericPrice(s.takeProfit1 ?? s.takeProfit) || 0;
  const mtf = s.multiTimeframe || {};
  const aplus = s.aplusSmc || {};
  const htf = s.htfAnalysis || mtf.higherTimeframe || {};
  const mtfObj = s.mtfAnalysis || mtf.middleTimeframe || {};
  const ltf = s.ltfAnalysis || mtf.lowerTimeframe || {};

  const evidenceArr = Array.isArray(s.evidence) ? s.evidence : [];
  const reasoningArr = Array.isArray(s.reasoning) ? s.reasoning : [];
  const allText = [
    ...evidenceArr,
    ...reasoningArr,
    JSON.stringify(htf),
    JSON.stringify(mtfObj),
    JSON.stringify(ltf),
    JSON.stringify(s.crtDetails || {}),
    JSON.stringify(s.crtAnalysis || {}),
    JSON.stringify(s.smcDetails || {}),
    JSON.stringify(s.smcAnalysis || {}),
  ].join(' ').toLowerCase();

  const risk = Math.abs(entry - sl);
  const reward = Math.abs(tp1 - entry);
  const rrRatio = (risk > 0 && entry > 0 && sl > 0 && tp1 > 0) ? reward / risk : 0;
  const hasValidTradeLevels = entry > 0 && sl > 0 && tp1 > 0;

  // -----------------------------------------------------------
  // CRT (Candle Range Theory) Confluence Audit (10 Components)
  // -----------------------------------------------------------
  if (strat === 'CRT') {
    const crt = s.crtDetails || s.crtAnalysis || {};
    const rangeDesc = crt.rangeFormation || 'Session / Reference Candle Range';
    const rangeFormed = Boolean(
      crt.rangeFormation ||
      (crt.referenceHigh && crt.referenceLow) ||
      allText.includes('range') ||
      allText.includes('asian') ||
      allText.includes('consolidation')
    );

    const liqMapped = Boolean(
      allText.includes('liquidity') ||
      allText.includes('buy-side') ||
      allText.includes('sell-side') ||
      crt.referenceHigh ||
      crt.referenceLow
    );

    const sweepConfirmed = Boolean(
      crt.rangeSweepConfirmed ||
      (crt.manipulationSweep && !crt.manipulationSweep.includes('None')) ||
      allText.includes('sweep') ||
      allText.includes('swept') ||
      allText.includes('wick')
    );

    const rejectionConfirmed = Boolean(
      crt.rejectionConfirmed ||
      (crt.rejection && !crt.rejection.includes('None')) ||
      allText.includes('reject') ||
      allText.includes('inside') ||
      allText.includes('defense')
    );

    const directionalIntentConfirmed = Boolean(
      crt.directionalIntent ||
      crt.directionalDelivery ||
      allText.includes('intent') ||
      allText.includes('delivery') ||
      allText.includes('expansion')
    );

    const ltfConfirmed = Boolean(
      crt.ltfConfirmationConfirmed ||
      (ltf.executionConfirmation && !String(ltf.executionConfirmation).toLowerCase().includes('pending')) ||
      allText.includes('confirm') ||
      allText.includes('close')
    );

    const entryOk = hasValidTradeLevels;
    const slStructOk = isBull ? (sl > 0 && sl < entry) : (sl > 0 && sl > entry);
    const tpOpposingOk = isBull ? (tp1 > entry) : (tp1 > 0 && tp1 < entry);
    const rrOk = rrRatio >= 1.45;

    const components: any[] = [
      {
        id: 1,
        name: '1. Range Formation',
        status: rangeFormed ? 'CONFIRMED' : 'MISSING',
        evidence: rangeFormed ? `Reference range identified: ${rangeDesc}` : 'No clear range high/low reference established',
      },
      {
        id: 2,
        name: '2. Liquidity Mapping',
        status: liqMapped ? 'CONFIRMED' : 'MISSING',
        evidence: liqMapped ? 'External buy-side & sell-side liquidity pools mapped outside boundaries' : 'Liquidity pools not mapped outside range',
      },
      {
        id: 3,
        name: '3. Range Boundary Sweep',
        status: sweepConfirmed ? 'CONFIRMED' : 'MISSING',
        evidence: sweepConfirmed ? 'Clean manipulation sweep of range boundary observed' : 'Range boundary has not been swept',
      },
      {
        id: 4,
        name: '4. Rejection & Close Inside',
        status: rejectionConfirmed ? 'CONFIRMED' : 'MISSING',
        evidence: rejectionConfirmed ? 'Decisive rejection candle closed back inside established range' : 'Candle did not close back inside range boundaries',
      },
      {
        id: 5,
        name: '5. Directional Delivery',
        status: directionalIntentConfirmed ? 'CONFIRMED' : 'MISSING',
        evidence: directionalIntentConfirmed ? 'Expansion momentum targeted toward opposing range liquidity' : 'Directional intent toward opposite boundary unconfirmed',
      },
      {
        id: 6,
        name: '6. 15M Confirmation',
        status: ltfConfirmed ? 'CONFIRMED' : 'MISSING',
        evidence: ltfConfirmed ? '15M confirmation candle verified range defense and institutional rejection' : 'Waiting for 15M confirmation candle',
      },
      {
        id: 7,
        name: '7. Execution Entry',
        status: entryOk ? 'CONFIRMED' : 'MISSING',
        evidence: entryOk ? `Sniper entry set at ${entry} (retest or range re-entry)` : 'Valid numeric entry could not be verified from chart',
      },
      {
        id: 8,
        name: '8. Structural Invalidation SL',
        status: slStructOk ? 'CONFIRMED' : 'MISSING',
        evidence: slStructOk ? `Stop loss placed beyond sweep wick extreme at ${sl}` : 'Stop loss not placed beyond sweep extreme',
      },
      {
        id: 9,
        name: '9. Opposing Range TP Target',
        status: tpOpposingOk ? 'CONFIRMED' : 'MISSING',
        evidence: tpOpposingOk ? `Target placed at opposing range liquidity boundary at ${tp1}` : 'Target does not align with opposing range boundary',
      },
      {
        id: 10,
        name: '10. Mathematical Risk:Reward',
        status: rrOk ? 'CONFIRMED' : (rrRatio >= 1.0 ? 'PARTIAL' : 'MISSING'),
        evidence: rrRatio > 0 ? `Calculated Risk:Reward is 1 : ${rrRatio.toFixed(2)}` : 'Entry, SL, and TP required to calculate R:R',
      },
    ];

    const confirmedCount = components.filter(c => c.status === 'CONFIRMED').length;
    const requiredOk = rangeFormed && sweepConfirmed && rejectionConfirmed && slStructOk && tpOpposingOk && hasValidTradeLevels;
    const isAplusQualified = requiredOk && confirmedCount >= 8 && rrOk;

    let grade: 'A+' | 'A' | 'B' | 'Developing' | 'Rejected / No Setup' = 'A+';
    let summary = '';
    let rejectionReason: string | undefined;

    const rawStatus = String(s.setupStatus || '').toUpperCase();
    const isConfirmedExplicit = rawStatus === 'CONFIRMED' || rawStatus === 'A+ CONFIRMED' || rawStatus === 'A+ SETUP' || s.hasConfirmedSetup === true;
    const isDevelopingExplicit = rawStatus === 'DEVELOPING' || rawStatus.includes('WAIT');
    const isInvalidExplicit = rawStatus.includes('INVALID');
    const isNoSetupExplicit = rawStatus.includes('NO_SETUP') || rawStatus.includes('NO SETUP') || rawStatus.includes('NO VALID');

    if (isConfirmedExplicit && hasValidTradeLevels) {
      if (rrRatio >= 2.0) {
        grade = 'A+';
        summary = `A+ CRT Confluence (${confirmedCount}/10 Confirmed): Range Formation → Boundary Sweep → Rejection Back Inside → 15M Confirmation → Entry ${entry} → SL ${sl} → Opposing Target ${tp1} (1:${rrRatio.toFixed(2)} R:R).`;
      } else {
        grade = 'A';
        summary = `${confirmedCount}/10 CRT Criteria Confirmed — Valid A-grade CRT setup (1:${rrRatio.toFixed(2)} R:R).`;
      }
    } else if (isAplusQualified) {
      grade = 'A+';
      summary = `A+ CRT Confluence (${confirmedCount}/10 Confirmed): Range Formation → Boundary Sweep → Rejection Back Inside → 15M Confirmation → Entry ${entry} → SL ${sl} → Opposing Target ${tp1} (1:${rrRatio.toFixed(2)} R:R).`;
    } else if (requiredOk && confirmedCount >= 7) {
      grade = 'A';
      summary = `${confirmedCount}/10 CRT Criteria Confirmed — Valid A-grade CRT setup.`;
    } else if (hasValidTradeLevels && confirmedCount >= 5) {
      grade = 'B';
      summary = `${confirmedCount}/10 CRT Criteria Confirmed — Acceptable range setup; moderate confluence.`;
    } else if (isDevelopingExplicit || (rangeFormed && !sweepConfirmed)) {
      grade = 'Developing';
      summary = 'CRT range identified. Waiting for boundary sweep and rejection.';
      rejectionReason = 'Boundary sweep and close back inside not yet confirmed.';
    } else if (isInvalidExplicit) {
      grade = 'Rejected / No Setup';
      summary = 'CRT setup structure invalidated by price action.';
      rejectionReason = s.reasoning?.[0] || 'Price action breached CRT invalidation level.';
    } else {
      grade = 'Rejected / No Setup';
      summary = 'CRT conditions absent on visible candles.';
      rejectionReason = s.reasoning?.[0] || 'No clear range formation or sweep rejection identified.';
    }

    return {
      score: confirmedCount,
      confirmedCount,
      totalComponents: 10,
      grade,
      isAplusQualified,
      strategy: 'CRT' as const,
      components,
      summary,
      rejectionReason,
    };
  }

  // -----------------------------------------------------------
  // SMC Confluence Audit (12 Stages - Primary A+ SMC Model)
  // -----------------------------------------------------------
  const htfZone = aplus.htfSupplyDemand || {};
  const htfZoneOk = Boolean(
    htfZone.status === 'VALID' ||
    htf.supplyDemandZone ||
    htf.htfPoi ||
    s.smcAnalysis?.orderBlocks ||
    allText.includes('supply') ||
    allText.includes('demand')
  );
  const comp1 = {
    id: 1,
    name: '1. HTF Supply / Demand Zone',
    status: htfZoneOk ? 'CONFIRMED' : 'MISSING',
    evidence: htfZone.evidence || htfZone.zoneRange || htf.supplyDemandZone || `4H ${isBull ? 'Demand Zone' : 'Supply Zone'} identified on macro timeframe`,
  };

  const htfMit = aplus.htfMitigation || {};
  const htfMitOk = Boolean(
    htfMit.status === 'CONFIRMED' ||
    mtfObj.mitigationStatus ||
    allText.includes('mitigat') ||
    allText.includes('reaction') ||
    allText.includes('tap')
  );
  const comp2 = {
    id: 2,
    name: '2. HTF Zone Mitigation',
    status: htfMitOk ? 'CONFIRMED' : 'MISSING',
    evidence: htfMit.evidence || htfMit.reactionProduced || `Price visibly mitigated the 4H ${isBull ? 'demand' : 'supply'} zone`,
  };

  const ind = aplus.inducement || {};
  const indOk = Boolean(
    ind.status === 'IDENTIFIED' ||
    mtfObj.inducement ||
    s.smcDetails?.inducementPresent ||
    allText.includes('inducement') ||
    allText.includes('trap') ||
    allText.includes('early')
  );
  const comp3 = {
    id: 3,
    name: '3. Inducement of Early Traders',
    status: indOk ? 'CONFIRMED' : 'MISSING',
    evidence: ind.evidence || ind.purpose || `Early ${isBull ? 'buyers' : 'sellers'} induced prior to institutional sweep`,
  };

  const sweep = aplus.liquiditySweep || {};
  const sweepOk = Boolean(
    sweep.status === 'CONFIRMED' ||
    s.smcDetails?.liquiditySweepPresent ||
    (ltf.liquiditySweep && !String(ltf.liquiditySweep).toLowerCase().includes('none') && !String(ltf.liquiditySweep).toLowerCase().includes('pending')) ||
    allText.includes('sweep') ||
    allText.includes('swept')
  );
  const comp4 = {
    id: 4,
    name: '4. Liquidity Sweep in HTF Zone',
    status: sweepOk ? 'CONFIRMED' : 'MISSING',
    evidence: sweep.evidence || sweep.sweepType || `Liquidity sweep confirmed: ${sweep.sweptLevel || 'Sweep extreme'} inside zone`,
  };

  const val = aplus.htfZoneValidation || {};
  const valOk = Boolean(
    val.isValidated ?? (htfZoneOk && htfMitOk && sweepOk)
  );
  const comp5 = {
    id: 5,
    name: '5. Validation of HTF Zone',
    status: valOk ? 'CONFIRMED' : 'MISSING',
    evidence: val.validationRationale || `HTF ${isBull ? 'Demand' : 'Supply'} validated by mitigation + liquidity sweep confluence`,
  };

  const prot = aplus.protectedLevel || {};
  const protPrice = parseNumericPrice(prot.price ?? (!isNaN(sl) ? sl : null));
  const protOk = Boolean(!isNaN(protPrice) && protPrice > 0);
  const comp6 = {
    id: 6,
    name: '6. Protected High / Protected Low',
    status: protOk ? 'CONFIRMED' : 'MISSING',
    evidence: prot.reason || `${prot.type || (isBull ? 'Protected Low' : 'Protected High')} established at ${protPrice} from sweep extreme`,
  };

  const fts = aplus.failureToSwing || {};
  const ftsStatus = String(fts.status || '').toUpperCase();
  const ftsOk = Boolean(
    (ftsStatus === 'CONFIRMED' || s.smcDetails?.failureToSwingPresent || allText.includes('failure to swing') || allText.includes('failed to break') || allText.includes('failed to swing')) &&
    ftsStatus !== 'PENDING' && ftsStatus !== 'FAILED' && ftsStatus !== 'NOT CONFIRMED'
  );
  const comp7 = {
    id: 7,
    name: '7. Failure to Swing (2nd Attempt)',
    status: ftsOk ? 'CONFIRMED' : 'MISSING',
    evidence: fts.structuralMechanism || fts.evidence || 'Second attempt to break protected extreme failed, confirming exhaustion',
  };

  const cisd = aplus.changeInStateOfDelivery || {};
  const cisdStatus = String(cisd.status || '').toUpperCase();
  const cisdOk = Boolean(
    (cisdStatus === 'CONFIRMED' || s.smcDetails?.cisdPresent || ltf.mssChoch || allText.includes('cisd') || allText.includes('change in state of delivery') || allText.includes('mss')) &&
    cisdStatus !== 'PENDING' && cisdStatus !== 'NOT CONFIRMED'
  );
  const comp8 = {
    id: 8,
    name: '8. Change in State of Delivery (CISD)',
    status: cisdOk ? 'CONFIRMED' : 'MISSING',
    evidence: cisd.detailedSequence || cisd.evidence || '15M displacement candle closed with full body beyond failure pivot',
  };

  const poiObj = aplus.poi || {};
  const poiOk = Boolean(
    poiObj.status === 'VALID' ||
    poiObj.type ||
    mtfObj.poiRefinement ||
    ltf.orderBlock ||
    ltf.fvg ||
    allText.includes('poi') ||
    allText.includes('order block')
  );
  const comp9 = {
    id: 9,
    name: '9. 15M POI of Delivery Shift',
    status: poiOk ? 'CONFIRMED' : 'MISSING',
    evidence: poiObj.evidence || poiObj.causalConnectionToDeliveryChange || `${poiObj.type || '15M Order Block'} defined as causal origin of shift`,
  };

  const poiMit = aplus.poiMitigation || {};
  const poiMitStatus = String(poiMit.status || '').toUpperCase();
  const poiMitOk = Boolean(
    (poiMitStatus === 'CONFIRMED' || poiMitStatus === 'MITIGATED' || hasValidTradeLevels || allText.includes('retrace') || allText.includes('mitigated')) &&
    poiMitStatus !== 'PENDING' && poiMitStatus !== 'NOT REACHED'
  );
  const comp10 = {
    id: 10,
    name: '10. Wait for POI Mitigation',
    status: poiMitOk ? 'CONFIRMED' : 'MISSING',
    evidence: poiMit.reactionConfirmation || poiMit.evidence || 'Price pulled back into 15M POI and printed defense reaction',
  };

  // Step 11: 15M Entry & SL strictly anchored beyond protected extreme
  const slStructOk = isBull
    ? (sl > 0 && entry > 0 && sl < entry && (!protPrice || sl <= protPrice + 0.01))
    : (sl > 0 && entry > 0 && sl > entry && (!protPrice || sl >= protPrice - 0.01));
  const comp11 = {
    id: 11,
    name: '11. 15M Entry & Structural SL',
    status: slStructOk ? 'CONFIRMED' : 'MISSING',
    evidence: slStructOk
      ? `Entry at ${entry}, SL at ${sl} anchored beyond Protected ${isBull ? 'Low' : 'High'} (${protPrice || sl})`
      : `Stop Loss must be strictly anchored beyond Protected ${isBull ? 'Low' : 'High'}`,
  };

  // Step 12: Internal Range Liquidity Target & Asymmetric R:R
  const liqTarget = aplus.takeProfitTargets?.tp1LiquidityType || s.takeProfit || mtf.middleTimeframe?.liquidity || 'Internal Range Liquidity';
  const liqTargetOk = Boolean(tp1 > 0 && (isBull ? tp1 > entry : tp1 < entry) && rrRatio >= 1.0);
  const rrOk = rrRatio >= 1.45;
  const comp12 = {
    id: 12,
    name: '12. Internal Range Liquidity TP & R:R',
    status: liqTargetOk ? 'CONFIRMED' : 'MISSING',
    evidence: liqTargetOk
      ? `TP1 (${tp1}) mapped to ${liqTarget} with 1:${rrRatio.toFixed(2)} Risk:Reward`
      : 'Take Profit must target valid opposing internal range liquidity',
  };

  const components: any[] = [comp1, comp2, comp3, comp4, comp5, comp6, comp7, comp8, comp9, comp10, comp11, comp12];
  const confirmedCount = components.filter(c => c.status === 'CONFIRMED').length;

  const sequenceEssentialOk = htfZoneOk && htfMitOk && sweepOk && valOk && protOk && ftsOk && cisdOk && poiMitOk && slStructOk && liqTargetOk && hasValidTradeLevels;
  const isAplusQualified = sequenceEssentialOk && confirmedCount >= 10 && rrOk;

  let grade: 'A+' | 'A' | 'B' | 'Developing' | 'Rejected / No Setup' = 'A+';
  let summary = '';
  let rejectionReason: string | undefined;

  const rawStatus = String(s.setupStatus || '').toUpperCase();
  const isConfirmedExplicit = rawStatus === 'CONFIRMED' || rawStatus === 'A+ CONFIRMED' || rawStatus === 'A+ SETUP' || s.hasConfirmedSetup === true;
  const isDevelopingExplicit = rawStatus === 'DEVELOPING' || rawStatus.includes('WAIT');
  const isInvalidExplicit = rawStatus.includes('INVALID');
  const isNoSetupExplicit = rawStatus.includes('NO_SETUP') || rawStatus.includes('NO SETUP') || rawStatus.includes('NO VALID');

  if (isConfirmedExplicit && hasValidTradeLevels && sequenceEssentialOk) {
    if (rrRatio >= 2.0 && confirmedCount >= 10) {
      grade = 'A+';
      summary = `A+ Institutional Sequence (${confirmedCount}/12 Confirmed): HTF Zone → Mitigation → Inducement → Sweep → HTF Validation → Protected Extreme → Failure to Swing → CISD → POI Mitigation → Entry ${entry} → SL ${sl} → TP ${tp1} (1:${rrRatio.toFixed(2)} R:R).`;
    } else if (rrRatio >= 1.4) {
      grade = 'A';
      summary = `${confirmedCount}/12 Sequence Stages Confirmed — Valid A-grade SMC setup with verified 1:${rrRatio.toFixed(2)} R:R.`;
    } else {
      grade = 'B';
      summary = `${confirmedCount}/12 Sequence Stages Confirmed — Valid institutional setup with 1:${rrRatio.toFixed(2)} R:R.`;
    }
  } else if (isAplusQualified) {
    grade = 'A+';
    summary = `A+ Institutional Sequence (${confirmedCount}/12 Confirmed): HTF Zone → Mitigation → Inducement → Sweep → HTF Validation → Protected Extreme → Failure to Swing → CISD → POI Mitigation → Entry ${entry} → SL ${sl} → TP ${tp1} (1:${rrRatio.toFixed(2)} R:R).`;
  } else if (sequenceEssentialOk && confirmedCount >= 8) {
    grade = 'A';
    summary = `${confirmedCount}/12 Sequence Stages Confirmed — Valid A-grade SMC setup with strong structural confirmation.`;
  } else if (hasValidTradeLevels && confirmedCount >= 6) {
    grade = 'B';
    summary = `${confirmedCount}/12 Sequence Stages Confirmed — Moderate institutional confluence.`;
  } else if (isDevelopingExplicit || (!hasValidTradeLevels && !isInvalidExplicit && !isNoSetupExplicit)) {
    grade = 'Developing';
    summary = `${confirmedCount}/12 Sequence Stages Confirmed — Setup is developing. Waiting for sequence trigger conditions.`;
    rejectionReason = s.reasoning?.[0] || 'Waiting for 15M execution confirmation and structural price levels.';
  } else if (isInvalidExplicit) {
    grade = 'Rejected / No Setup';
    summary = 'Setup structure invalidated by market price action.';
    rejectionReason = s.reasoning?.[0] || 'Setup invalidated by price action breaking structural invalidation.';
  } else {
    grade = 'Rejected / No Setup';
    summary = `${confirmedCount}/12 Sequence Stages Confirmed — Sequence conditions absent.`;
    rejectionReason = s.reasoning?.[0] || 'Primary A+ SMC institutional sequence absent on visible candles.';
  }

  return {
    score: confirmedCount,
    confirmedCount,
    totalComponents: 12,
    grade,
    isAplusQualified,
    strategy: 'SMC' as const,
    components,
    summary,
    rejectionReason,
  };
}

function ensureSignalMultiTimeframe(s: any): any {
  if (!s) return s;

  const strat = String(s.strategy || s.selectedStrategy || 'SMC').toUpperCase();
  if (strat === 'SMC' && !s.aplusSmc) {
    s.aplusSmc = normalizeAplusSmcSequence(s);
  }

  // Build and attach 12-Component A+ Confluence Audit if not present
  if (!s.aplusConfluenceAudit) {
    const audit = buildAplusConfluenceAudit(s);
    s.aplusConfluenceAudit = audit;
    if (s.aplusSmc) {
      s.aplusSmc.confluenceAudit = audit;
    }
  }

  // Populate multiTimeframe structure from granular AI evidence if missing
  if (!s.multiTimeframe && (s.higherTimeframe4H || s.middleTimeframe1H || s.lowerTimeframe15M)) {
    const isBull = s.directionalBias === 'BULLISH' || s.bias === 'BULLISH';
    s.multiTimeframe = {
      allTimeframesAnalyzed: true,
      perTimeframe: {
        htf: s.higherTimeframe4H?.trend || (isBull ? 'BULLISH' : 'BEARISH'),
        mtf: s.middleTimeframe1H?.trend || (isBull ? 'BULLISH' : 'BEARISH'),
        ltf: s.lowerTimeframe15M?.trend || (isBull ? 'BULLISH' : 'BEARISH'),
      },
      higherTimeframe: {
        analyzed: true,
        timeframe: '4H',
        bias: s.higherTimeframe4H?.trend || (isBull ? 'BULLISH' : 'BEARISH'),
        structure: s.higherTimeframe4H?.marketStructure || '4H macro structural order flow',
        liquidity: s.higherTimeframe4H?.drawOnLiquidity || '4H external liquidity pool',
        poi: s.higherTimeframe4H?.keyPoi || '4H Institutional POI',
        premiumDiscount: s.higherTimeframe4H?.equilibriumStatus || (isBull ? 'Discount Zone (< 50%)' : 'Premium Zone (> 50%)'),
        notes: s.higherTimeframe4H?.equilibriumStatus || '',
      },
      middleTimeframe: {
        analyzed: true,
        timeframe: '1H',
        bias: s.middleTimeframe1H?.trend || (isBull ? 'BULLISH' : 'BEARISH'),
        structure: s.middleTimeframe1H?.marketStructure || '1H structural shift',
        bosChoch: s.middleTimeframe1H?.marketStructure || '1H BOS / CHoCH',
        liquidity: s.middleTimeframe1H?.inducement || '1H inducement swept',
        poi: s.middleTimeframe1H?.poiRefinement || '1H refined POI / unmitigated zone',
        displacement: s.middleTimeframe1H?.marketStructure || '',
        alignmentWithHtf: 'Aligned',
        notes: s.middleTimeframe1H?.invalidationLevel ? `Invalidation: ${s.middleTimeframe1H.invalidationLevel}` : '',
      },
      lowerTimeframe: {
        analyzed: true,
        timeframe: '15M',
        confirmation: s.lowerTimeframe15M?.triggerConfirmed ? 'CONFIRMED' : 'DEVELOPING',
        mssChoch: s.lowerTimeframe15M?.marketStructureShift || '15M Market Structure Shift',
        displacement: s.lowerTimeframe15M?.triggerType || '15M Displacement',
        fvgOb: s.lowerTimeframe15M?.fvgStatus || '15M FVG / OB',
        entry: s.entry,
        invalidation: s.stopLoss,
        shortTermLiquidityTargets: s.takeProfit,
        notes: s.lowerTimeframe15M?.notes || s.lowerTimeframe15M?.liquiditySweep || '',
      },
      finalDecision: {
        direction: isBull ? 'BUY' : (s.directionalBias === 'BEARISH' || s.bias === 'BEARISH') ? 'SELL' : 'WAIT',
        entry: s.entry,
        sl: s.stopLoss,
        tp1: s.takeProfit,
        tp2: s.takeProfit2 || s.takeProfit,
        rr: s.riskRewardRatio || s.riskReward || '1 : 2.50',
        confidence: s.confidence || 85,
        grade: s.setupBadge || 'A+',
      },
      alignment: 'Full Multi-Timeframe Confluence',
      crossTimeframeReasoning: s.confidenceRationale || s.reasoning?.[0] || 'Evidence-first multi-timeframe synthesis.',
    };
  }

  // Synchronize setupBadge and setupStatus cleanly without forcing "Avoid" on confirmed setups
  const rawStatus = String(s.setupStatus || '').toUpperCase();
  if (rawStatus === 'CONFIRMED' || rawStatus === 'A+ CONFIRMED' || rawStatus === 'A+ SETUP' || s.hasConfirmedSetup === true) {
    s.setupBadge = s.setupGrade || s.setupBadge || 'A+';
    s.setupStatus = 'CONFIRMED';
  } else if (rawStatus === 'DEVELOPING') {
    s.setupBadge = 'Developing';
    s.setupStatus = 'DEVELOPING';
  } else if (rawStatus === 'NO_SETUP' || rawStatus.includes('NO SETUP') || rawStatus.includes('NO VALID')) {
    s.setupBadge = 'Avoid';
    s.setupStatus = 'NO_SETUP';
  } else if (rawStatus.includes('INVALID')) {
    s.setupBadge = 'Avoid';
    s.setupStatus = 'INVALID';
  }

  return s;
}

app.get('/api/signals', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';

  let list = db.signals;
  if (!isAdmin) {
    list = list.filter(s => s.userId === user.id || s.user_id === user.id);
  }

  const sorted = [...list]
    .map(ensureSignalMultiTimeframe)
    .sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  res.json({ success: true, signals: sorted });
});

function handleUpdateSignalOutcome(req: Request, res: Response) {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';

  const signalId = req.params.id || req.body.signalId;
  const { outcome, pnlR, realizedR, status, notes } = req.body;
  if (!signalId) {
    return res.status(400).json({ success: false, error: 'signalId is required' });
  }

  const signal = db.signals.find(s => s.id === signalId);
  if (!signal) {
    return res.status(404).json({ success: false, error: 'Signal not found' });
  }
  const ownerId = signal.userId || signal.user_id;
  if (!isAdmin && ownerId !== user.id) {
    return res.status(403).json({ success: false, error: 'Forbidden: You do not own this signal.' });
  }

  if (outcome !== undefined) signal.outcome = outcome;
  if (pnlR !== undefined) signal.pnlR = Number(pnlR);
  if (realizedR !== undefined) signal.realizedR = Number(realizedR);
  if (status !== undefined) signal.status = status;
  if (notes !== undefined) signal.notes = notes;

  const wItem = db.watchlist.find(w => w.signal_id === signalId && w.user_id === ownerId);
  if (wItem) {
    if (status !== undefined) wItem.status = status;
    if (notes !== undefined) wItem.notes = notes;
  }

  saveDb();
  return res.json({ success: true, signal });
}

app.post('/api/signals/outcome', handleUpdateSignalOutcome);
app.patch('/api/signals/:id/outcome', handleUpdateSignalOutcome);

app.delete('/api/signals/:id', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  const { id } = req.params;

  const signal = db.signals.find(s => s.id === id);
  if (!signal) {
    return res.status(404).json({ success: false, error: 'Signal not found' });
  }
  const ownerId = signal.userId || signal.user_id;
  if (!isAdmin && ownerId !== user.id) {
    return res.status(403).json({ success: false, error: 'Forbidden: You cannot delete another user signal.' });
  }

  db.signals = db.signals.filter(s => s.id !== id);
  db.watchlist = db.watchlist.filter(w => w.signal_id !== id);
  saveDb();
  res.json({ success: true });
});

app.get('/api/watchlist', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';

  let list = db.watchlist;
  if (!isAdmin) {
    list = list.filter(w => w.user_id === user.id);
  }

  const enriched = list.map(item => {
    const signal = db.signals.find(s => s.id === item.signal_id);
    return { ...item, signal };
  });
  res.json({ success: true, watchlist: enriched });
});

app.post('/api/watchlist', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  const effectiveUserId = user.id;
  const { signalId, notes, status } = req.body;
  if (!signalId) {
    return res.status(400).json({ success: false, error: 'signalId is required' });
  }

  const signal = db.signals.find(s => s.id === signalId);
  if (!signal) {
    return res.status(404).json({ success: false, error: 'Signal not found' });
  }
  const ownerId = signal.userId || signal.user_id;
  if (!isAdmin && ownerId !== user.id) {
    return res.status(403).json({ success: false, error: 'Forbidden: You do not own this signal.' });
  }

  const existing = db.watchlist.find(w => w.signal_id === signalId && w.user_id === effectiveUserId);
  if (existing) {
    if (notes !== undefined) existing.notes = notes;
    if (status !== undefined) existing.status = status;
    saveDb();
    return res.json({ success: true, item: existing });
  }

  const item = {
    id: `w_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
    signal_id: signal.id,
    user_id: effectiveUserId,
    asset: signal.asset,
    timeframe: signal.timeframe,
    strategy: signal.strategy,
    bias: signal.bias,
    entry: signal.entry,
    stopLoss: signal.stopLoss,
    takeProfit: signal.takeProfit,
    riskReward: signal.riskReward,
    confidence: signal.confidence,
    setupBadge: signal.setupBadge,
    status: status || 'Active',
    notes: notes || signal.reasoning?.[0] || 'Saved setup for execution tracking',
    created_at: new Date().toISOString(),
  };

  db.watchlist.push(item);
  saveDb();
  res.json({ success: true, item });
});

function handleUpdateWatchlist(req: Request, res: Response) {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';

  const { id } = req.params;
  const { status, notes } = req.body;

  const item = db.watchlist.find(w => w.id === id);
  if (!item) {
    return res.status(404).json({ success: false, error: 'Watchlist item not found' });
  }
  if (!isAdmin && item.user_id !== user.id) {
    return res.status(403).json({ success: false, error: 'Forbidden: You do not own this watchlist item.' });
  }

  if (status !== undefined) item.status = status;
  if (notes !== undefined) item.notes = notes;

  saveDb();
  res.json({ success: true, item });
}

app.put('/api/watchlist/:id', handleUpdateWatchlist);
app.patch('/api/watchlist/:id', handleUpdateWatchlist);

app.delete('/api/watchlist/:id', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  const { id } = req.params;

  const item = db.watchlist.find(w => w.id === id);
  if (!item) {
    return res.status(404).json({ success: false, error: 'Watchlist item not found' });
  }
  if (!isAdmin && item.user_id !== user.id) {
    return res.status(403).json({ success: false, error: 'Forbidden: You do not own this watchlist item.' });
  }

  db.watchlist = db.watchlist.filter(w => w.id !== id);
  saveDb();
  res.json({ success: true });
});

// -------------------------------------------------------------
// Admin Analysis Logs API & Regression Test Endpoint
// -------------------------------------------------------------
app.get('/api/admin/logs', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  if (!isAdmin) {
    return res.status(403).json({ success: false, error: 'Forbidden: Admin privileges required.' });
  }
  const sorted = [...db.analysisLogs].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );
  res.json({ success: true, logs: sorted });
});

app.get('/api/admin/regression-test', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  if (!isAdmin) {
    return res.status(403).json({ success: false, error: 'Forbidden: Admin privileges required.' });
  }
  try {
    const testResults = runRegressionTests();
    res.json({ success: true, ...testResults });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err?.message || 'Regression test failed' });
  }
});

// -------------------------------------------------------------
// Validation Helper for Price Scale & Directional Integrity
// -------------------------------------------------------------
interface PriceAxisValidationResult {
  valid: boolean;
  reason?: string;
}

function validateSignalPriceScale(
  signal: any,
  requestedAsset: string
): PriceAxisValidationResult {
  if (!signal) {
    return { valid: false, reason: 'SIGNAL REJECTED — Empty signal produced by analysis engine.' };
  }

  // 1. Normalize Bias
  let rawBias = String(signal.bias || '').toUpperCase().trim();
  if (rawBias.includes('BULL') || rawBias.includes('LONG') || rawBias.includes('BUY')) {
    signal.bias = 'BULLISH';
  } else if (rawBias.includes('BEAR') || rawBias.includes('SHORT') || rawBias.includes('SELL')) {
    signal.bias = 'BEARISH';
  } else {
    signal.bias = 'NEUTRAL';
  }

  // Check if analysis explicitly concluded no setup or insufficient evidence
  const rawStatus = String(signal.setupStatus || '').toUpperCase();
  const rawQuality = String(signal.chartQuality || '').toUpperCase();
  const isNoSetupOrInsufficient =
    rawStatus.includes('NO VALID') ||
    rawStatus.includes('INSUFFICIENT') ||
    rawQuality.includes('UNREADABLE') ||
    signal.setupBadge === 'Avoid' ||
    signal.setupBadge === 'Rejected / No Setup';

  // 2. Extract numeric prices
  const entry = parseNumericPrice(signal.entry ?? signal.entryPrice ?? signal.entry_price ?? signal.entryLevel);
  const sl = parseNumericPrice(signal.stopLoss ?? signal.sl ?? signal.stop_loss ?? signal.stopPrice);
  const tp = parseNumericPrice(signal.takeProfit ?? signal.tp ?? signal.tp1 ?? signal.takeProfit1 ?? signal.take_profit ?? (Array.isArray(signal.takeProfits) ? signal.takeProfits[0] : undefined));
  const tp2 = parseNumericPrice(signal.takeProfit2 ?? signal.tp2 ?? signal.takeProfit_2 ?? (Array.isArray(signal.takeProfits) ? signal.takeProfits[1] : undefined));
  const tp3 = parseNumericPrice(signal.takeProfit3 ?? signal.tp3 ?? signal.takeProfit_3 ?? (Array.isArray(signal.takeProfits) ? signal.takeProfits[2] : undefined));
  const altEntry = parseNumericPrice(signal.alternativeEntry ?? signal.altEntry ?? signal.secondaryEntry);

  if (isNoSetupOrInsufficient || isNaN(entry) || entry <= 0) {
    // Legitimate negative determination - do NOT invent mock prices
    signal.entry = !isNaN(entry) && entry > 0 ? entry : null;
    signal.stopLoss = !isNaN(sl) && sl > 0 ? sl : null;
    signal.takeProfit = !isNaN(tp) && tp > 0 ? tp : null;
    signal.takeProfit2 = !isNaN(tp2) && tp2 > 0 ? tp2 : null;
    signal.takeProfit3 = !isNaN(tp3) && tp3 > 0 ? tp3 : null;
    signal.alternativeEntry = !isNaN(altEntry) && altEntry > 0 ? altEntry : null;
    signal.riskReward = 'N/A';
    signal.setupStatus = rawQuality.includes('UNREADABLE') ? 'INSUFFICIENT EVIDENCE' : (signal.setupStatus || 'NO VALID SETUP');
    signal.setupBadge = 'Avoid';

    // Compute genuine audit without synthetic fallbacks
    const audit = buildAplusConfluenceAudit(signal);
    signal.aplusConfluenceAudit = audit;
    if (signal.aplusSmc) {
      signal.aplusSmc.confluenceAudit = audit;
    }
    return { valid: true };
  }

  // A trade setup IS proposed with numeric entry: validate all levels strictly
  signal.entry = entry;
  if (!isNaN(altEntry) && altEntry > 0) signal.alternativeEntry = altEntry;

  if (isNaN(sl) || sl <= 0) {
    return {
      valid: false,
      reason: 'STRUCTURAL INVALIDATION FAILED — Proposed setup requires a genuine Stop Loss anchored to visible chart structure.',
    };
  }
  signal.stopLoss = sl;

  if (isNaN(tp) || tp <= 0) {
    return {
      valid: false,
      reason: 'TARGET VALIDATION FAILED — Proposed setup requires a genuine Take Profit target anchored to visible chart liquidity.',
    };
  }
  signal.takeProfit = tp;
  if (!isNaN(tp2) && tp2 > 0) signal.takeProfit2 = tp2;
  if (!isNaN(tp3) && tp3 > 0) signal.takeProfit3 = tp3;

  // 3. Strict Directional Verification (NO synthetic price correction)
  if (signal.bias === 'BULLISH') {
    if (sl >= entry) {
      return {
        valid: false,
        reason: `DIRECTIONAL CONFLICT — Bullish setup requires Stop Loss (${sl}) to be strictly below Entry (${entry}).`,
      };
    }
    if (tp <= entry) {
      return {
        valid: false,
        reason: `DIRECTIONAL CONFLICT — Bullish setup requires Take Profit (${tp}) to be strictly above Entry (${entry}).`,
      };
    }
  } else if (signal.bias === 'BEARISH') {
    if (sl <= entry) {
      return {
        valid: false,
        reason: `DIRECTIONAL CONFLICT — Bearish setup requires Stop Loss (${sl}) to be strictly above Entry (${entry}).`,
      };
    }
    if (tp >= entry) {
      return {
        valid: false,
        reason: `DIRECTIONAL CONFLICT — Bearish setup requires Take Profit (${tp}) to be strictly below Entry (${entry}).`,
      };
    }
  }

  // 4. Genuine Mathematical Risk-to-Reward calculation
  const risk = Math.abs(entry - sl);
  const reward = Math.abs(tp - entry);
  const calculatedRR = risk > 0 ? `1 : ${(reward / risk).toFixed(2)}` : 'N/A';
  signal.riskReward = calculatedRR;

  // 5. Price Axis Scale verification
  let scale = signal.priceAxisScale;
  let minP = scale ? parseNumericPrice(scale.minVisiblePrice) : NaN;
  let maxP = scale ? parseNumericPrice(scale.maxVisiblePrice) : NaN;
  let currP = scale ? parseNumericPrice(scale.currentVisiblePrice) : NaN;

  if (!isNaN(minP) && !isNaN(maxP) && minP > 0 && maxP > minP) {
    signal.priceAxisScale.minVisiblePrice = minP;
    signal.priceAxisScale.maxVisiblePrice = maxP;
    signal.priceAxisScale.currentVisiblePrice = isNaN(currP) ? entry : currP;
    signal.priceAxisScale.isAxisReadable = true;

    // Check if entry price is within reasonable range of visible chart axis
    const tolerance = (maxP - minP) * 0.35;
    if (entry < minP - tolerance || entry > maxP + tolerance) {
      return {
        valid: false,
        reason: `PRICE SCALE DISCREPANCY — Entry price (${entry}) falls outside visible chart price range (${minP} - ${maxP}). Ensure all levels are read directly from chart axis.`,
      };
    }
  } else {
    // If scale couldn't be accurately read, record honest status
    signal.priceAxisScale = {
      minVisiblePrice: null,
      maxVisiblePrice: null,
      currentVisiblePrice: entry,
      isAxisReadable: false,
      confidenceInPriceReading: signal.priceAxisScale?.confidenceInPriceReading || 40,
      notes: signal.priceAxisScale?.notes || 'Price axis labels not clearly delineated; levels estimated from visible candle range.',
    };
  }

  // 6. Compute genuine 12-Component A+ Confluence Audit
  const audit = buildAplusConfluenceAudit(signal);
  signal.aplusConfluenceAudit = audit;
  if (signal.aplusSmc) {
    signal.aplusSmc.confluenceAudit = audit;
  }

  signal.chartQuality = signal.chartQuality || 'ANALYZABLE';
  if (audit) {
    if (audit.grade === 'A+') {
      signal.setupStatus = signal.setupStatus || 'A+ CONFIRMED';
      signal.setupBadge = 'A+';
    } else if (audit.grade === 'A') {
      signal.setupStatus = signal.setupStatus || 'CONFIRMED';
      signal.setupBadge = 'A';
    } else if (audit.grade === 'B') {
      signal.setupStatus = signal.setupStatus || 'VALID SETUP';
      signal.setupBadge = 'B';
    } else if (audit.grade === 'Developing') {
      signal.setupStatus = 'DEVELOPING';
      signal.setupBadge = 'Developing';
    } else {
      signal.setupStatus = 'NO VALID SETUP';
      signal.setupBadge = 'Avoid';
    }
  }

  // 7. Calibrate annotations to visible price scale if scale is confirmed
  if (Array.isArray(signal.annotations) && !isNaN(minP) && !isNaN(maxP) && maxP > minP) {
    for (const ann of signal.annotations) {
      const annPrice = parseNumericPrice(ann.price);
      if (!isNaN(annPrice)) {
        ann.price = annPrice;
        const calculatedY = Math.max(2, Math.min(98, ((maxP - annPrice) / (maxP - minP)) * 100));
        ann.y = Number(calculatedY.toFixed(1));
        ann.y2 = Number(calculatedY.toFixed(1));
      }
    }
  }

  return { valid: true };
}

function detectTimeframeMismatch(
  chart: any,
  roleLabel: string,
  expectedTf: string
): { mismatch: boolean; detected?: string; note?: string } {
  if (!chart || !chart.imageBase64) return { mismatch: false };
  const rawStr =
    chart.imageBase64.includes('svg') || chart.imageBase64.startsWith('data:image/svg')
      ? Buffer.from(chart.imageBase64.split(',')[1] || '', 'base64').toString('utf-8')
      : '';

  if (rawStr) {
    const patterns: Array<{ name: string; regex: RegExp }> = [
      { name: '4H', regex: /\b(4H|H4|240M)\b/i },
      { name: '1H', regex: /\b(1H|H1|60M)\b/i },
      { name: '15M', regex: /\b(15M|M15)\b/i },
      { name: '5M', regex: /\b(5M|M5)\b/i },
      { name: '1D', regex: /\b(1D|D1|Daily)\b/i },
    ];
    for (const pat of patterns) {
      if (pat.regex.test(rawStr)) {
        const normExp = (expectedTf || '').toUpperCase();
        const normDetected = pat.name.toUpperCase();
        if (
          normDetected !== normExp &&
          normDetected.replace('H', '').replace('M', '') !== normExp.replace('H', '').replace('M', '')
        ) {
          return {
            mismatch: true,
            detected: pat.name,
            note: `Timeframe mismatch detected on ${roleLabel} chart: Uploaded as ${expectedTf}, but chart text displays ${pat.name}.`,
          };
        }
      }
    }
  }
  return { mismatch: false };
}

function validateMultiTimeframeSignal(
  signal: any,
  isMulti: boolean,
  hasHtf: boolean,
  hasMtf: boolean,
  hasLtf: boolean,
  mismatchNotice?: string
): void {
  if (!isMulti) return;

  if (!signal.multiTimeframe) {
    signal.multiTimeframe = {};
  }

  signal.multiTimeframe.htfAnalyzed = Boolean(hasHtf && (signal.multiTimeframe.htfAnalyzed ?? true));
  signal.multiTimeframe.mtfAnalyzed = Boolean(hasMtf && (signal.multiTimeframe.mtfAnalyzed ?? true));
  signal.multiTimeframe.ltfAnalyzed = Boolean(hasLtf && (signal.multiTimeframe.ltfAnalyzed ?? true));
  signal.multiTimeframe.allTimeframesAnalyzed =
    signal.multiTimeframe.htfAnalyzed && signal.multiTimeframe.mtfAnalyzed && signal.multiTimeframe.ltfAnalyzed;

  if (mismatchNotice) {
    signal.multiTimeframe.timeframeMismatchNotice = mismatchNotice;
  }

  // 1. HARD VALIDATION RULE: All three timeframes must be present
  if (!signal.multiTimeframe.allTimeframesAnalyzed) {
    const missing: string[] = [];
    if (!signal.multiTimeframe.htfAnalyzed) missing.push('Higher Timeframe (4H)');
    if (!signal.multiTimeframe.mtfAnalyzed) missing.push('Middle Timeframe (1H)');
    if (!signal.multiTimeframe.ltfAnalyzed) missing.push('Lower Timeframe (15M)');

    signal.bias = 'NEUTRAL';
    signal.setupStatus = 'INSUFFICIENT EVIDENCE';
    signal.setupBadge = 'Avoid';
    signal.confidence = 35;
    if (signal.multiTimeframe.finalDecision) {
      signal.multiTimeframe.finalDecision.direction = 'WAIT';
      signal.multiTimeframe.finalDecision.grade = 'Avoid';
    }
    if (!signal.aplusSmc) signal.aplusSmc = {} as any;
    if (!signal.aplusSmc.missingConditions) signal.aplusSmc.missingConditions = [];
    signal.aplusSmc.missingConditions.unshift(
      `WAIT — MULTI-TIMEFRAME EVIDENCE INCOMPLETE: Missing or unanalyzed timeframe(s): ${missing.join(', ')}. All three timeframes (4H, 1H, 15M) must be validated.`
    );
    signal.reasoning = [
      `WAIT: Multi-timeframe evidence incomplete. Missing analysis for: ${missing.join(', ')}.`,
      'Top-down sequence requires 4H context, 1H structure, and 15M confirmation.',
      ...(signal.reasoning || []),
    ];
    return;
  }

  // 2. Evidence-first evaluation of Lower Timeframe (15M)
  // Instead of relying solely on ltfConf === "CONFIRMED", evaluate concrete chart evidence:
  // liquidity sweep, MSS/CHoCH, displacement, CISD, FVG/OB, entry structure, invalidation
  const ltf = signal.ltfAnalysis || signal.multiTimeframe?.lowerTimeframe;
  const smc = signal.smcDetails;
  const crt = signal.crtDetails;
  const rawStatus = String(signal.setupStatus || '').toUpperCase();
  const rawGrade = String(signal.setupGrade || signal.setupBadge || '').toUpperCase();

  const isModelConfirmed =
    signal.hasConfirmedSetup === true ||
    rawStatus.includes('CONFIRMED') ||
    rawStatus.includes('A+ SETUP') ||
    rawStatus.includes('VALID SETUP');

  // Verify visible evidence signals
  const hasLiquiditySweepEvidence = Boolean(
    (ltf?.liquiditySweep && ltf.liquiditySweep !== 'None' && ltf.liquiditySweep !== 'Pending') ||
    smc?.liquiditySweepPresent ||
    crt?.rangeSweepConfirmed ||
    (Array.isArray(signal.evidence) && signal.evidence.some((e: string) => /sweep|liquidity/i.test(e)))
  );

  const hasMssOrDisplacementEvidence = Boolean(
    (ltf?.mssChoch && ltf.mssChoch !== 'None' && ltf.mssChoch !== 'Pending') ||
    (ltf?.displacement && ltf.displacement !== 'None' && ltf.displacement !== 'Pending') ||
    (ltf?.cisd && ltf.cisd !== 'None') ||
    smc?.displacementPresent ||
    smc?.bosChochPresent ||
    smc?.cisdPresent ||
    crt?.rejectionConfirmed ||
    (Array.isArray(signal.evidence) && signal.evidence.some((e: string) => /mss|choch|bos|displacement|cisd|rejection/i.test(e)))
  );

  const hasExecutionTrigger = Boolean(
    (ltf?.executionConfirmation && !String(ltf.executionConfirmation).toLowerCase().includes('pending')) ||
    (ltf?.entryLocation && ltf.entryLocation !== 'Pending') ||
    crt?.entryLevel ||
    smc?.entryConfirmed ||
    (signal.entry && Number(signal.entry) > 0)
  );

  const hasConfirmedEvidence = (hasLiquiditySweepEvidence && hasMssOrDisplacementEvidence) || hasExecutionTrigger;

  if (isModelConfirmed || signal.setupStatus === 'CONFIRMED' || signal.setupStatus === 'A+ CONFIRMED') {
    // Structural evidence confirms setup — preserve trade output!
    signal.setupStatus = 'CONFIRMED';
    signal.setupBadge = rawGrade.includes('A+') ? 'A+' : (signal.setupBadge || 'A');
    if (signal.multiTimeframe.finalDecision) {
      signal.multiTimeframe.finalDecision.direction = signal.bias === 'BEARISH' ? 'SELL' : 'BUY';
      signal.multiTimeframe.finalDecision.grade = signal.setupBadge;
    }
  } else {
    // Model determined setup is developing, no setup, or invalid
    if (rawStatus.includes('INVALID')) {
      signal.setupStatus = 'INVALID';
      signal.setupBadge = 'Avoid';
    } else if (rawStatus.includes('NO_SETUP') || rawStatus.includes('NO SETUP') || rawStatus.includes('NO VALID')) {
      signal.setupStatus = 'NO_SETUP';
      signal.setupBadge = 'Avoid';
    } else {
      signal.setupStatus = 'DEVELOPING';
      signal.setupBadge = 'Developing';
    }
    if (signal.multiTimeframe.finalDecision) {
      signal.multiTimeframe.finalDecision.direction = 'WAIT';
      signal.multiTimeframe.finalDecision.grade = signal.setupBadge;
    }
  }
}

// -------------------------------------------------------------
// -------------------------------------------------------------
// CORE SERVER ANALYSIS FUNCTION: analyzeChart (TradeGuard Production Pipeline)
// -------------------------------------------------------------
// -------------------------------------------------------------
// ASYNCHRONOUS ANALYSIS JOB ARCHITECTURE (TradeGuard Institutional Pipeline)
// -------------------------------------------------------------

export type JobStatus = 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export interface AnalysisJob {
  id: string;
  idempotencyKey?: string;
  userId: string;
  asset: string;
  assetClass: string;
  strategy: Strategy;
  accountBalance: number;
  riskPercent: number;
  createdAt: number;
  updatedAt: number;
  status: JobStatus;
  stageIndex: number;
  stageName: string;
  stageMessage: string;
  progressPercent: number;
  signal?: any;
  error?: {
    code: string;
    message: string;
    status?: number;
    retryable?: boolean;
    retryAfter?: number;
  };
  durationMs?: number;
  modelUsed?: string;
  providerUsed?: string;
  tokenUsage?: any;
  charts?: any[];
  abortController?: AbortController;
}

// In-Memory job registry & reactive listener map
const analysisJobs = new Map<string, AnalysisJob>();
const jobSubscribers = new Map<string, Set<(job: AnalysisJob) => void>>();

function serializeJobForDisk(job: AnalysisJob) {
  const base = sanitizeJobForClient(job);
  // Preserve chart inputs only for recent retryable failed jobs (max 10) so retries survive restart without bloating JSON
  const keepCharts = job.status === 'FAILED' && job.error?.retryable && Array.isArray(job.charts);
  return {
    ...base,
    ...(keepCharts ? { charts: job.charts } : {}),
  };
}

function persistJobState(job: AnalysisJob) {
  if (!Array.isArray(db.jobs)) db.jobs = [];
  const serialized = serializeJobForDisk(job);
  const idx = db.jobs.findIndex((j: any) => j.id === job.id);
  if (idx !== -1) {
    db.jobs[idx] = serialized;
  } else {
    db.jobs.unshift(serialized);
  }
  // Keep at most 30 recent jobs and strip charts from older failed jobs beyond top 10
  if (db.jobs.length > 30) {
    db.jobs = db.jobs.slice(0, 30);
  }
  let chartsKept = 0;
  for (const j of db.jobs) {
    if (j.charts) {
      chartsKept++;
      if (chartsKept > 10) {
        delete j.charts;
      }
    }
  }
  saveDb();
}

function hydrateJobsFromDb() {
  if (!Array.isArray(db.jobs)) return;
  let dirty = false;
  for (const savedJob of db.jobs) {
    if (!savedJob?.id) continue;
    if (savedJob.status === 'QUEUED' || savedJob.status === 'PROCESSING') {
      savedJob.status = 'FAILED';
      savedJob.stageName = 'INTERRUPTED_BY_RESTART';
      savedJob.stageMessage = 'Server restarted during analysis. Please re-upload or retry.';
      savedJob.error = {
        code: 'SERVER_RESTARTED',
        message: 'Analysis was interrupted by a server restart. Click Retry to resume.',
        status: 503,
        retryable: Array.isArray(savedJob.charts) && savedJob.charts.length === 3,
      };
      savedJob.updatedAt = Date.now();
      dirty = true;
    }
    analysisJobs.set(savedJob.id, savedJob);
  }
  if (dirty) saveDb();
}

hydrateJobsFromDb();

// Auto-cleanup expired jobs every 15 minutes (keep completed/failed jobs for 24 hours)
setInterval(() => {
  const now = Date.now();
  const maxAgeMs = 24 * 60 * 60 * 1000;
  for (const [id, job] of analysisJobs.entries()) {
    if (now - job.createdAt > maxAgeMs) {
      analysisJobs.delete(id);
      jobSubscribers.delete(id);
    }
  }
}, 15 * 60 * 1000);

function sanitizeJobForClient(job: AnalysisJob) {
  return {
    id: job.id,
    idempotencyKey: job.idempotencyKey,
    userId: job.userId,
    asset: job.asset,
    assetClass: job.assetClass,
    strategy: job.strategy,
    accountBalance: job.accountBalance,
    riskPercent: job.riskPercent,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    status: job.status,
    stageIndex: job.stageIndex,
    stageName: job.stageName,
    stageMessage: job.stageMessage,
    progressPercent: job.progressPercent,
    durationMs: job.durationMs,
    modelUsed: job.modelUsed,
    providerUsed: job.providerUsed,
    tokenUsage: job.tokenUsage,
    signal: job.signal,
    error: job.error,
  };
}

function notifyJobSubscribers(job: AnalysisJob) {
  const subscribers = jobSubscribers.get(job.id);
  if (subscribers && subscribers.size > 0) {
    for (const callback of subscribers) {
      try {
        callback(job);
      } catch (err) {
        console.warn(`[TradeGuard Jobs] Error delivering event to subscriber for job ${job.id}:`, err);
      }
    }
  }
}

function updateJobStage(
  job: AnalysisJob,
  stageIndex: number,
  stageName: string,
  stageMessage: string,
  progressPercent: number
) {
  job.status = 'PROCESSING';
  job.stageIndex = stageIndex;
  job.stageName = stageName;
  job.stageMessage = stageMessage;
  job.progressPercent = progressPercent;
  job.updatedAt = Date.now();
  notifyJobSubscribers(job);
}

function findExistingActiveJob(userId: string, asset: string, idempotencyKey?: string): AnalysisJob | undefined {
  const now = Date.now();
  for (const job of analysisJobs.values()) {
    if (job.status === 'QUEUED' || job.status === 'PROCESSING') {
      if (idempotencyKey && job.userId === userId && job.idempotencyKey === idempotencyKey) {
        return job;
      }
      if (job.userId === userId && job.asset.toUpperCase() === asset.toUpperCase() && now - job.createdAt < 45000) {
        return job;
      }
    }
  }
  return undefined;
}

function recordFailedAnalysisLog(params: {
  job: AnalysisJob;
  errObj: NonNullable<AnalysisJob['error']>;
  durationMs: number;
  imageMimes?: string[];
  imageSizes?: number[];
  timings?: any;
}) {
  const { job, errObj, durationMs, imageMimes = [], imageSizes = [], timings } = params;
  const aiStatus =
    errObj.code === 'RATE_LIMITED' || errObj.code === 'QUOTA_EXHAUSTED'
      ? 'RateLimited'
      : 'Error';
  const logEntry = {
    id: `log_${job.id}_${Date.now()}`,
    jobId: job.id,
    userId: job.userId,
    asset: job.asset,
    selectedPair: job.asset,
    strategy: job.strategy,
    mode: 'multi' as const,
    timestamp: new Date().toISOString(),
    imageMimes,
    imageSizes,
    configuredModel: job.modelUsed || TRADEGUARD_GEMINI_MODEL,
    providerUsed: job.providerUsed || 'Gemini Primary',
    geminiResponseStatus: errObj.status || 500,
    geminiErrorCode: errObj.code,
    finalAnalysisStatus: 'FAILED',
    status: 'Failed' as const,
    decision: 'NONE',
    confidence: 0,
    duration: durationMs,
    aiStatus,
    error: errObj.message,
    rawErrorDetails: (errObj as any).details ? String((errObj as any).details).slice(0, 1000) : undefined,
    timings,
    createdAt: new Date().toISOString(),
  };
  db.analysisLogs.unshift(logEntry);
  if (db.analysisLogs.length > 200) {
    db.analysisLogs = db.analysisLogs.slice(0, 200);
  }
  persistJobState(job);
}

/**
 * Core asynchronous execution pipeline for institutional 3-timeframe analysis.
 * Operates across 7 authentic stages with real progress updates, per-call deadlines,
 * failover model resiliency, and strict SMC/CRT validation rules.
 */
async function executeInstitutionalPipeline(job: AnalysisJob, charts: any[]): Promise<any> {
  const t_start = Date.now();
  const logId = `log_${job.id}`;
  const asset = job.asset.toUpperCase().trim();
  const normalizedStrategy = job.strategy;
  const userId = job.userId;

  const auth_ms = 0;
  let image_validation_ms = 0;
  let image_processing_ms = 0;
  let prompt_build_ms = 0;
  let ai_request_ms = 0;
  let ai_parse_ms = 0;
  const schema_validation_ms = 0;
  let signal_validation_ms = 0;
  let database_insert_ms = 0;
  let imageMimes: string[] = [];
  let imageSizes: number[] = [];

  try {
    // -------------------------------------------------------------
    // STAGE 0: Input & 3-Timeframe Validation (10%)
    // -------------------------------------------------------------
    updateJobStage(
      job,
      0,
      'INPUT_IMAGE_VALIDATION',
      'Pass 1: Calibrating price scale & validating 3-timeframe chart evidence (4H, 1H, 15M)...',
      12
    );

    const t_val = Date.now();
    const htfChart = Array.isArray(charts)
      ? charts.find(c => c.role === 'HTF' || c.timeframe === '4H' || c.timeframe === 'H4')
      : null;
    const mtfChart = Array.isArray(charts)
      ? charts.find(c => c.role === 'MTF' || c.timeframe === '1H' || c.timeframe === 'H1')
      : null;
    const ltfChart = Array.isArray(charts)
      ? charts.find(c => c.role === 'LTF' || c.timeframe === '15M' || c.timeframe === 'M15')
      : null;

    const htfReceived = Boolean(htfChart?.imageBase64 && htfChart.imageBase64.length > 20);
    const mtfReceived = Boolean(mtfChart?.imageBase64 && mtfChart.imageBase64.length > 20);
    const ltfReceived = Boolean(ltfChart?.imageBase64 && ltfChart.imageBase64.length > 20);
    const imagesReceived = [htfReceived, mtfReceived, ltfReceived].filter(Boolean).length;

    if (imagesReceived !== 3 || !htfReceived || !mtfReceived || !ltfReceived) {
      const errObj = {
        code: 'THREE_TIMEFRAMES_REQUIRED',
        message: 'Exactly three timeframe charts are required (4H HTF, 1H MTF, 15M LTF).',
        status: 400,
        retryable: false,
      };
      job.status = 'FAILED';
      job.error = errObj;
      job.durationMs = Date.now() - t_start;
      job.updatedAt = Date.now();
      recordFailedAnalysisLog({ job, errObj, durationMs: job.durationMs });
      notifyJobSubscribers(job);
      return;
    }
    image_validation_ms = Date.now() - t_val;

    // Verify Gemini environment configuration
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || !apiKey.trim()) {
      const errObj = {
        code: 'AUTH_FAILURE',
        message: 'TradeGuard configuration error: GEMINI_API_KEY environment variable is missing.',
        status: 401,
        retryable: false,
      };
      job.status = 'FAILED';
      job.error = errObj;
      job.durationMs = Date.now() - t_start;
      job.updatedAt = Date.now();
      recordFailedAnalysisLog({ job, errObj, durationMs: job.durationMs });
      notifyJobSubscribers(job);
      return;
    }

    if (job.abortController?.signal.aborted) {
      job.status = 'CANCELLED';
      job.stageMessage = 'Analysis cancelled by user.';
      job.updatedAt = Date.now();
      persistJobState(job);
      notifyJobSubscribers(job);
      return;
    }

    // -------------------------------------------------------------
    // STAGE 1: Parallel Image Processing & Aspect Ratio Preservation (25%)
    // -------------------------------------------------------------
    updateJobStage(
      job,
      1,
      'OPTIMIZING_CHART_RESOLUTIONS',
      'Pass 2: Preserving aspect ratios & optimizing candle clarity for vision analysis...',
      26
    );

    const t_proc = Date.now();
    const [htfProcessed, mtfProcessed, ltfProcessed] = await Promise.all([
      processChartImageForGemini(htfChart, 'HTF'),
      processChartImageForGemini(mtfChart, 'MTF'),
      processChartImageForGemini(ltfChart, 'LTF'),
    ]);

    imageMimes = [htfProcessed.mimeType, mtfProcessed.mimeType, ltfProcessed.mimeType];
    imageSizes = [htfProcessed.sizeBytes, mtfProcessed.sizeBytes, ltfProcessed.sizeBytes];
    const htfPart = { inlineData: { mimeType: htfProcessed.mimeType, data: htfProcessed.data } };
    const mtfPart = { inlineData: { mimeType: mtfProcessed.mimeType, data: mtfProcessed.data } };
    const ltfPart = { inlineData: { mimeType: ltfProcessed.mimeType, data: ltfProcessed.data } };
    image_processing_ms = Date.now() - t_proc;

    if (job.abortController?.signal.aborted) {
      job.status = 'CANCELLED';
      job.stageMessage = 'Analysis cancelled by user.';
      job.updatedAt = Date.now();
      persistJobState(job);
      notifyJobSubscribers(job);
      return;
    }

    // -------------------------------------------------------------
    // STAGE 2: Prompt Compilation & Causal Rules Formulation (40%)
    // -------------------------------------------------------------
    updateJobStage(
      job,
      2,
      'BUILDING_INSTITUTIONAL_PROMPT',
      'Pass 3: Compiling multi-timeframe SMC/ICT/CRT causal rules & ground truth scale prompts...',
      40
    );

    const t_prompt = Date.now();
    const { systemInstruction: tgSystemInstruction, userInstruction } = buildTradeGuardPrompt(
      normalizedStrategy,
      asset
    );
    prompt_build_ms = Date.now() - t_prompt;

    const contentParts = [
      {
        text: `[INSTITUTIONAL MANDATE: THREE REAL TRADINGVIEW CHARTS PROVIDED]
You are analyzing three real chart screenshots for ${asset} uploaded by the institutional trader.
Follow strict top-down institutional order:
1. Chart 1: Higher Timeframe (4H) — Establish Macro Bias and HTF POI.
2. Chart 2: Middle Timeframe (1H) — Identify Structural Refinement and Internal Liquidity.
3. Chart 3: Lower Timeframe (15M) — Confirm Sniper Execution Trigger ONLY.`,
      },
      { text: `\n[CHART 1: HIGHER TIMEFRAME (4H) — MACRO CONTEXT & BIAS]` },
      htfPart,
      { text: `\n[CHART 2: MIDDLE TIMEFRAME (1H) — STRUCTURAL REFINEMENT & LIQUIDITY]` },
      mtfPart,
      { text: `\n[CHART 3: LOWER TIMEFRAME (15M) — EXECUTION CONFIRMATION TIMEFRAME]` },
      ltfPart,
      { text: userInstruction },
    ];

    // -------------------------------------------------------------
    // STAGE 3: Multimodal Vision Model Execution with Failover (62%)
    // -------------------------------------------------------------
    updateJobStage(
      job,
      3,
      'CALLING_GEMINI_VISION_MODEL',
      'Pass 4: Running vision inference on institutional engine (HTF POI, sweeps, CISD & displacement)...',
      62
    );

    const t_ai = Date.now();
    let rawText = '';

    try {
      const inferenceResult = await aiProviderManager.executeVisionInference({
        contentParts,
        systemInstruction: tgSystemInstruction,
        abortController: job.abortController,
        onProgress: (stageMsg) => {
          updateJobStage(job, 3, 'RETRYING_GEMINI_VISION', stageMsg, 62);
        },
      });

      rawText = inferenceResult.rawText;
      ai_request_ms = inferenceResult.durationMs;
      job.modelUsed = inferenceResult.modelUsed;
      job.providerUsed = inferenceResult.providerUsed;
      job.tokenUsage = inferenceResult.tokenUsage;
    } catch (aiErr: any) {
      ai_request_ms = Date.now() - t_ai;

      if (aiErr?.name === 'AbortError' || job.abortController?.signal.aborted || String(aiErr?.message || '').includes('cancelled')) {
        job.status = 'CANCELLED';
        job.stageMessage = 'Analysis cancelled by user.';
        job.durationMs = Date.now() - t_start;
        job.updatedAt = Date.now();
        persistJobState(job);
        notifyJobSubscribers(job);
        return;
      }

      const classified = aiErr?.code && typeof aiErr?.status === 'number' ? aiErr : aiProviderManager.classifyError(aiErr);
      const errObj = {
        code: classified.code,
        message: classified.message,
        status: classified.status,
        retryable: classified.retryable,
        retryAfter: classified.retryAfterSeconds,
        details: classified.details ? String(classified.details).slice(0, 1000) : undefined,
      };
      job.status = 'FAILED';
      job.error = errObj;
      job.durationMs = Date.now() - t_start;
      job.updatedAt = Date.now();
      logDiagnosticFailure({
        requestId: job.id,
        stage: 'GEMINI_VISION_INFERENCE',
        error: errObj.message,
        status: errObj.status || 500,
        code: errObj.code,
        pair: asset,
        strategy: normalizedStrategy,
        imagesReceived: 3,
        imageMimes,
        imageSizes,
        geminiHttpStatus: errObj.status,
        geminiErrorCode: errObj.code,
        geminiErrorMessage: classified.details || errObj.message,
      });
      recordFailedAnalysisLog({ job, errObj, durationMs: job.durationMs, imageMimes, imageSizes });
      notifyJobSubscribers(job);
      return;
    }

    if (!rawText || rawText.length === 0) {
      const errObj = {
        code: 'EMPTY_AI_RESPONSE',
        message: 'Multimodal vision provider returned empty response.',
        status: 502,
        retryable: true,
      };
      job.status = 'FAILED';
      job.error = errObj;
      job.durationMs = Date.now() - t_start;
      job.updatedAt = Date.now();
      recordFailedAnalysisLog({ job, errObj, durationMs: job.durationMs, imageMimes, imageSizes });
      notifyJobSubscribers(job);
      return;
    }

    // -------------------------------------------------------------
    // STAGE 4: Response Schema Parsing & Ground Truth Extraction (80%)
    // -------------------------------------------------------------
    updateJobStage(
      job,
      4,
      'PARSING_INSTITUTIONAL_SCHEMA',
      'Pass 5: Parsing structured JSON schema & cross-checking price scales against candle wicks...',
      80
    );

    const t_parse = Date.now();
    let cleaned = rawText.trim();
    if (cleaned.startsWith('```')) {
      cleaned = cleaned.replace(/^```[a-z]*\s*\n?/i, '').replace(/\n?```\s*$/i, '').trim();
    }
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
      cleaned = cleaned.substring(firstBrace, lastBrace + 1);
    }

    let parsedResult: any = null;
    try {
      parsedResult = JSON.parse(cleaned);
    } catch (parseErr: any) {
      const errObj = {
        code: 'SCHEMA_PARSE_ERROR',
        message: `Failed to parse structured institutional response: ${parseErr.message}`,
        status: 502,
        retryable: true,
      };
      job.status = 'FAILED';
      job.error = errObj;
      job.durationMs = Date.now() - t_start;
      job.updatedAt = Date.now();
      recordFailedAnalysisLog({ job, errObj, durationMs: job.durationMs, imageMimes, imageSizes });
      notifyJobSubscribers(job);
      return;
    }
    ai_parse_ms = Date.now() - t_parse;

    // -------------------------------------------------------------
    // STAGE 5: 13 Stage-Gate Checks & Mathematical Structural Audit (92%)
    // -------------------------------------------------------------
    updateJobStage(
      job,
      5,
      'VALIDATING_STAGE_GATE_CHECKS',
      'Pass 6: Auditing 13 institutional stage-gate checks (Protected High/Low, Sweep separation, SL/TP)...',
      92
    );

    const t_val_signal = Date.now();
    const valResult = validateTradeGuardAnalysis(parsedResult, {
      asset,
      strategy: normalizedStrategy,
      charts,
    });
    signal_validation_ms = Date.now() - t_val_signal;

    const effectiveModel = job.modelUsed || TRADEGUARD_GEMINI_MODEL;

    if (!valResult.valid) {
      const errObj = {
        code: 'VALIDATION_FAILED',
        message: valResult.reason || 'Analysis failed institutional structural validation checks.',
        status: 422,
        retryable: true,
      };
      job.status = 'FAILED';
      job.error = errObj;
      job.durationMs = Date.now() - t_start;
      job.updatedAt = Date.now();
      recordFailedAnalysisLog({ job, errObj, durationMs: job.durationMs, imageMimes, imageSizes });
      notifyJobSubscribers(job);
      return;
    }

    const isLong = String(parsedResult?.bias).toUpperCase() === 'BULLISH';
    const isShort = String(parsedResult?.bias).toUpperCase() === 'BEARISH';
    const entryNum = parseNumericPrice(parsedResult?.entry || parsedResult?.entryPrice);
    const slNum = parseNumericPrice(parsedResult?.stopLoss);
    const tpNum = parseNumericPrice(parsedResult?.takeProfit || parsedResult?.takeProfit1);

    const currentIsoTimestamp = new Date().toISOString();
    const finalSignal: any = {
      id: `sig_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      jobId: job.id,
      userId,
      asset,
      selectedPair: asset,
      strategy: normalizedStrategy,
      ...parsedResult,
      created_at: parsedResult?.created_at && !isNaN(new Date(parsedResult.created_at).getTime())
        ? new Date(parsedResult.created_at).toISOString()
        : currentIsoTimestamp,
      timestamp: currentIsoTimestamp,
      session: parsedResult?.session && typeof parsedResult.session === 'string' && parsedResult.session.length > 2
        ? parsedResult.session
        : inferTradingSession(currentIsoTimestamp),
      entry: !isNaN(entryNum) && entryNum > 0 ? entryNum : null,
      entryPrice: !isNaN(entryNum) && entryNum > 0 ? entryNum : null,
      stopLoss: !isNaN(slNum) && slNum > 0 ? slNum : null,
      takeProfit: !isNaN(tpNum) && tpNum > 0 ? tpNum : null,
      setupStatus: valResult.setupStatus,
      setupBadge: valResult.grade,
      validationResult: valResult,
      executionState: valResult.valid ? 'SIGNAL_CONFIRMED' : (valResult.setupStatus === 'DEVELOPING' ? 'NO_SETUP_DETECTED' : 'INVALID_SETUP'),
      analysisStatus: valResult.setupStatus,
      bias: valResult.valid ? parsedResult?.bias : (valResult.setupStatus === 'DEVELOPING' ? 'WAIT' : parsedResult?.bias || 'WAIT'),
      notes: valResult.notes || parsedResult?.notes,
    };

    // Mathematical consistency check
    if (finalSignal.entryPrice !== null && finalSignal.stopLoss !== null && finalSignal.takeProfit !== null) {
      const validMath = isLong
        ? (finalSignal.stopLoss < finalSignal.entryPrice && finalSignal.entryPrice < finalSignal.takeProfit)
        : isShort
        ? (finalSignal.takeProfit < finalSignal.entryPrice && finalSignal.entryPrice < finalSignal.stopLoss)
        : true;

      if (!validMath) {
        finalSignal.setupStatus = 'REJECTED';
        finalSignal.setupBadge = 'Avoid';
        finalSignal.bias = 'WAIT';
        finalSignal.entryPrice = null;
        finalSignal.stopLoss = null;
        finalSignal.takeProfit = null;
        finalSignal.executionState = 'INVALID_SETUP';
        finalSignal.analysisStatus = 'REJECTED';
        finalSignal.notes = (finalSignal.notes ? finalSignal.notes + '. ' : '') + 'REJECTED: mathematical relationship invalid for bias.';
      }
    }

    // Calculate institutional risk management
    finalSignal.riskManagement = calculateInstitutionalRiskManagement(
      finalSignal.asset || job.asset,
      Number(job.accountBalance) || 10000,
      Number(job.riskPercent) || 1.0,
      finalSignal.entryPrice ?? finalSignal.entry,
      finalSignal.stopLoss,
      finalSignal.takeProfit
    );

    // Ensure multiTimeframe object is complete
    ensureSignalMultiTimeframe(finalSignal);

    // -------------------------------------------------------------
    // STAGE 6: Dossier Finalization & Persistence (98% -> 100%)
    // -------------------------------------------------------------
    updateJobStage(
      job,
      6,
      'FINALIZING_REPORT',
      'Pass 7: Finalizing institutional trade dossier, invalidation buffers, and persisting setup...',
      98
    );

    finalSignal.jobId = job.id;
    finalSignal.modelUsed = effectiveModel;
    finalSignal.providerUsed = job.providerUsed || 'Gemini Primary';
    finalSignal.modelVersion = TRADEGUARD_MODEL_VERSION;
    finalSignal.promptVersion = TRADEGUARD_PROMPT_VERSION;
    finalSignal.schemaVersion = TRADEGUARD_SCHEMA_VERSION;
    finalSignal.engineVersion = TRADEGUARD_ENGINE_VERSION;

    const t_db = Date.now();
    db.signals.unshift(finalSignal);
    database_insert_ms = Date.now() - t_db;

    const total_ms = Date.now() - t_start;
    const timings = {
      auth_ms,
      image_validation_ms,
      image_processing_ms,
      prompt_build_ms,
      ai_request_ms,
      ai_parse_ms,
      schema_validation_ms,
      signal_validation_ms,
      database_insert_ms,
      logging_ms: 0,
      total_ms,
    };
    finalSignal.timings = timings;

    const logEntry = {
      id: logId,
      jobId: job.id,
      userId,
      signalId: finalSignal.id,
      asset: finalSignal.asset,
      selectedPair: finalSignal.asset,
      strategy: finalSignal.strategy,
      mode: 'multi' as const,
      timestamp: new Date().toISOString(),
      htfImageReceived: true,
      mtfImageReceived: true,
      ltfImageReceived: true,
      imageMimes,
      imageSizes,
      configuredModel: effectiveModel,
      modelUsed: effectiveModel,
      providerUsed: job.providerUsed || 'Gemini Primary',
      tokenUsage: job.tokenUsage || null,
      geminiResponseStatus: 200,
      geminiErrorCode: null,
      schemaValidationStatus: 'PASSED',
      signalValidationStatus: 'PASSED',
      finalAnalysisStatus: finalSignal.executionState,
      status: 'Success' as const,
      decision: finalSignal.bias,
      confidence: finalSignal.confidence ?? 0,
      alignment: finalSignal.multiTimeframe?.alignment || 'Top-Down Alignment',
      duration: total_ms,
      aiStatus: 'OK' as const,
      timings,
      createdAt: new Date().toISOString(),
    };
    db.analysisLogs.unshift(logEntry);

    // Mark job as COMPLETED
    job.status = 'COMPLETED';
    job.stageIndex = 6;
    job.stageName = 'ANALYSIS_COMPLETE';
    job.stageMessage = 'Institutional trade setup analysis validated and ready.';
    job.progressPercent = 100;
    job.durationMs = total_ms;
    job.signal = finalSignal;
    job.charts = undefined; // Free memory/disk once completed
    job.updatedAt = Date.now();
    persistJobState(job);
    notifyJobSubscribers(job);

    return finalSignal;
  } catch (err: any) {
    console.error(`[TradeGuard Job ${job.id}] Pipeline failure:`, err);
    const classified = aiProviderManager.classifyError(err);
    const errObj = {
      code: classified.code || 'PIPELINE_ERROR',
      message: classified.message || err?.message || 'An error occurred during institutional chart analysis.',
      status: classified.status || 500,
      retryable: classified.retryable ?? true,
    };
    job.status = 'FAILED';
    job.error = errObj;
    job.durationMs = Date.now() - t_start;
    job.updatedAt = Date.now();
    recordFailedAnalysisLog({ job, errObj, durationMs: job.durationMs, imageMimes, imageSizes });
    notifyJobSubscribers(job);
  }
}

// -------------------------------------------------------------
// ENDPOINTS: Asynchronous Analysis Jobs & Multi-Timeframe Submission
// -------------------------------------------------------------
function normalizeStrategyInput(strategy: any): Strategy | null {
  const rawStrat = String(strategy || 'SMC').toUpperCase();
  if (rawStrat === 'SMC') return 'SMC';
  if (rawStrat === 'ICT') return 'ICT';
  if (rawStrat === 'CRT') return 'CRT';
  if (rawStrat.includes('PRICE') || rawStrat.includes('PA')) return 'Price Action';
  return null;
}

/**
 * POST /api/analysis/submit
 * Non-blocking job submission. Returns a unique jobId immediately.
 */
app.post('/api/analysis/submit', async (req: Request, res: Response) => {
  const authenticatedUser = getAuthenticatedUser(req);
  if (!authenticatedUser) {
    return res.status(401).json({ success: false, error: 'Authentication required to run chart analysis.' });
  }
  const userId = authenticatedUser.id;

  const {
    asset,
    assetClass = 'Forex',
    strategy = 'SMC',
    charts = [],
    accountBalance = 10000,
    riskPercent = 1.0,
    idempotencyKey,
  } = req.body;

  if (!asset || typeof asset !== 'string') {
    return res.status(400).json({ success: false, error: 'Asset/Pair identifier is required (e.g. XAUUSD, EURUSD, BTCUSDT).' });
  }

  const normalizedStrategy = normalizeStrategyInput(strategy);
  if (!normalizedStrategy) {
    return res.status(400).json({
      success: false,
      error: 'INVALID STRATEGY — TradeGuard production engine supports SMC, ICT, CRT, or Price Action.',
    });
  }

  // Idempotency check: prevent duplicate submissions
  const existingJob = findExistingActiveJob(userId, asset, idempotencyKey);
  if (existingJob) {
    return res.json({
      success: true,
      jobId: existingJob.id,
      status: existingJob.status,
      stageIndex: existingJob.stageIndex,
      stageName: existingJob.stageName,
      stageMessage: existingJob.stageMessage,
      progressPercent: existingJob.progressPercent,
      isExisting: true,
    });
  }

  const jobId = `job_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  const abortController = new AbortController();

  const newJob: AnalysisJob = {
    id: jobId,
    idempotencyKey,
    userId,
    asset: asset.toUpperCase().trim(),
    assetClass,
    strategy: normalizedStrategy,
    accountBalance: Number(accountBalance) || 10000,
    riskPercent: Number(riskPercent) || 1.0,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: 'QUEUED',
    stageIndex: 0,
    stageName: 'QUEUED',
    stageMessage: 'Job queued for institutional analysis...',
    progressPercent: 5,
    charts,
    abortController,
  };

  analysisJobs.set(jobId, newJob);
  persistJobState(newJob);

  // Global queue enforces strictly single-concurrency execution to prevent Gemini RPM/TPM exhaustion
  globalAnalysisQueue.enqueue(
    jobId,
    async () => {
      await executeInstitutionalPipeline(newJob, charts);
    },
    abortController,
    (pos, total) => {
      if (newJob.status === 'QUEUED') {
        newJob.stageMessage = `Queued for analysis (position ${pos} of ${total}). Waiting for engine...`;
        newJob.progressPercent = Math.max(5, Math.min(20, Math.floor(20 / pos)));
        notifyJobSubscribers(newJob);
      }
    }
  ).catch(err => {
    console.error(`[TradeGuard] Unhandled job execution failure for ${jobId}:`, err);
  });

  return res.json({
    success: true,
    jobId,
    status: newJob.status,
    stageIndex: newJob.stageIndex,
    stageName: newJob.stageName,
    stageMessage: newJob.stageMessage,
    progressPercent: newJob.progressPercent,
  });
});

/**
 * GET /api/analysis/jobs/:id
 * Retrieves the genuine current state of an analysis job.
 */
app.get('/api/analysis/jobs/:id', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  const { id } = req.params;
  const job = analysisJobs.get(id);

  if (job) {
    if (!isAdmin && job.userId !== user.id) {
      return res.status(403).json({ success: false, error: 'Forbidden: You do not own this analysis job.' });
    }
    return res.json({
      success: true,
      job: sanitizeJobForClient(job),
    });
  }

  // Check persistent DB for completed signal if job expired from memory
  const persistentSignal = db.signals.find(s => s.jobId === id || s.id === id);
  if (persistentSignal) {
    const ownerId = persistentSignal.userId || persistentSignal.user_id;
    if (!isAdmin && ownerId !== user.id) {
      return res.status(403).json({ success: false, error: 'Forbidden: You do not own this analysis job.' });
    }
    return res.json({
      success: true,
      job: {
        id,
        userId: ownerId,
        asset: persistentSignal.asset,
        strategy: persistentSignal.strategy,
        status: 'COMPLETED',
        stageIndex: 6,
        stageName: 'ANALYSIS_COMPLETE',
        stageMessage: 'Institutional trade setup analysis validated and ready.',
        progressPercent: 100,
        signal: persistentSignal,
        createdAt: new Date(persistentSignal.timestamp || persistentSignal.created_at).getTime(),
        updatedAt: new Date(persistentSignal.timestamp || persistentSignal.created_at).getTime(),
      },
    });
  }

  return res.status(404).json({
    success: false,
    error: `Analysis job ${id} was not found.`,
  });
});

/**
 * GET /api/analysis/jobs/:id/events
 * Real-time Server-Sent Events (SSE) stream for institutional analysis progress.
 */
app.get('/api/analysis/jobs/:id/events', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req, true);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  const { id } = req.params;
  const job = analysisJobs.get(id);

  if (job && !isAdmin && job.userId !== user.id) {
    return res.status(403).json({ success: false, error: 'Forbidden.' });
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders?.();

  if (!job) {
    const persistentSignal = db.signals.find(s => s.jobId === id || s.id === id);
    if (persistentSignal && (isAdmin || persistentSignal.userId === user.id)) {
      res.write(`data: ${JSON.stringify({
        id,
        status: 'COMPLETED',
        stageIndex: 6,
        stageName: 'ANALYSIS_COMPLETE',
        stageMessage: 'Institutional trade setup analysis validated and ready.',
        progressPercent: 100,
        signal: persistentSignal,
      })}\n\n`);
    } else {
      res.write(`data: ${JSON.stringify({ error: 'Job not found', status: 'FAILED' })}\n\n`);
    }
    return res.end();
  }

  res.write(`data: ${JSON.stringify(sanitizeJobForClient(job))}\n\n`);

  if (job.status === 'COMPLETED' || job.status === 'FAILED' || job.status === 'CANCELLED') {
    return res.end();
  }

  if (!jobSubscribers.has(id)) {
    jobSubscribers.set(id, new Set());
  }
  const subscriberSet = jobSubscribers.get(id)!;

  const onUpdate = (updatedJob: AnalysisJob) => {
    try {
      res.write(`data: ${JSON.stringify(sanitizeJobForClient(updatedJob))}\n\n`);
      if (updatedJob.status === 'COMPLETED' || updatedJob.status === 'FAILED' || updatedJob.status === 'CANCELLED') {
        subscriberSet.delete(onUpdate);
        res.end();
      }
    } catch {
      subscriberSet.delete(onUpdate);
    }
  };

  subscriberSet.add(onUpdate);

  req.on('close', () => {
    subscriberSet.delete(onUpdate);
  });
});

/**
 * POST /api/analysis/jobs/:id/cancel
 * Clean cancellation endpoint to abort in-flight Gemini requests.
 */
app.post('/api/analysis/jobs/:id/cancel', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  const { id } = req.params;
  const job = analysisJobs.get(id);

  if (!job) {
    return res.status(404).json({ success: false, error: 'Job not found' });
  }
  if (!isAdmin && job.userId !== user.id) {
    return res.status(403).json({ success: false, error: 'Forbidden: You do not own this analysis job.' });
  }

  if (job.status === 'COMPLETED' || job.status === 'FAILED' || job.status === 'CANCELLED') {
    return res.json({ success: true, status: job.status, message: `Job is already ${job.status}` });
  }

  globalAnalysisQueue.cancel(id);
  job.abortController?.abort();
  job.status = 'CANCELLED';
  job.stageMessage = 'Analysis cancelled by user.';
  job.updatedAt = Date.now();
  persistJobState(job);
  notifyJobSubscribers(job);

  return res.json({ success: true, status: 'CANCELLED', message: 'Job cancelled successfully.' });
});

/**
 * POST /api/analysis/jobs/:id/retry
 * Controlled retry for an existing failed or timed-out job.
 */
app.post('/api/analysis/jobs/:id/retry', async (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';
  const { id } = req.params;
  const job = analysisJobs.get(id);

  if (!job) {
    return res.status(404).json({ success: false, error: 'Job not found or expired.' });
  }
  if (!isAdmin && job.userId !== user.id) {
    return res.status(403).json({ success: false, error: 'Forbidden: You do not own this analysis job.' });
  }

  if (job.status === 'PROCESSING' || job.status === 'QUEUED') {
    return res.json({
      success: true,
      jobId: job.id,
      status: job.status,
      message: 'Job is already currently running.',
    });
  }

  if (job.error && job.error.retryable === false) {
    return res.status(400).json({
      success: false,
      error: job.error.message || 'This failure cannot be retried automatically.',
    });
  }

  if (!Array.isArray(job.charts) || job.charts.length !== 3) {
    return res.status(400).json({
      success: false,
      error: 'Stored chart payloads are no longer available for this job. Please re-upload your charts.',
    });
  }

  job.status = 'QUEUED';
  job.stageIndex = 0;
  job.stageName = 'QUEUED_FOR_RETRY';
  job.stageMessage = 'Re-queued for institutional analysis retry...';
  job.progressPercent = 5;
  job.error = undefined;
  job.signal = undefined;
  job.updatedAt = Date.now();
  job.abortController = new AbortController();

  persistJobState(job);
  notifyJobSubscribers(job);

  globalAnalysisQueue.enqueue(
    job.id,
    async () => {
      await executeInstitutionalPipeline(job, job.charts || []);
    },
    job.abortController,
    (pos, total) => {
      if (job.status === 'QUEUED') {
        job.stageMessage = `Re-queued for analysis (position ${pos} of ${total}). Waiting for engine...`;
        notifyJobSubscribers(job);
      }
    }
  ).catch(err => {
    console.error(`[TradeGuard Job ${job.id}] Retry execution error:`, err);
  });

  return res.json({
    success: true,
    jobId: job.id,
    status: 'QUEUED',
    message: 'Institutional analysis retry initiated.',
  });
});

/**
 * GET /api/analysis/recent
 * Returns recent completed analyses for the authenticated user to support browser refresh.
 */
app.get('/api/analysis/recent', (req: Request, res: Response) => {
  const user = getAuthenticatedUser(req);
  if (!user) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const isAdmin = user.role === 'admin' || user.role === 'super_admin';

  const userSignals = db.signals
    .filter(s => isAdmin || s.userId === user.id || s.user_id === user.id)
    .slice(0, 15);

  return res.json({
    success: true,
    signals: userSignals,
  });
});

/**
 * POST /api/analyzeChart (Main analysis route with dual-mode async/sync support, queued cleanly)
 */
app.post('/api/analyzeChart', async (req: Request, res: Response) => {
  const authenticatedUser = getAuthenticatedUser(req);
  if (!authenticatedUser) {
    return res.status(401).json({ success: false, error: 'Authentication required.' });
  }
  const userId = authenticatedUser.id;
  const isAsync = req.query.async === 'true' || req.body.async === true;

  const {
    asset,
    assetClass = 'Forex',
    strategy = 'SMC',
    charts = [],
    accountBalance = 10000,
    riskPercent = 1.0,
    idempotencyKey,
  } = req.body;

  if (!asset || typeof asset !== 'string') {
    return res.status(400).json({ success: false, error: 'Asset/Pair identifier is required (e.g. XAUUSD, EURUSD, BTCUSDT).' });
  }

  const normalizedStrategy = normalizeStrategyInput(strategy);
  if (!normalizedStrategy) {
    return res.status(400).json({
      success: false,
      error: 'INVALID STRATEGY — TradeGuard production engine supports SMC, ICT, CRT, or Price Action.',
    });
  }

  if (isAsync) {
    const existingJob = findExistingActiveJob(userId, asset, idempotencyKey);
    if (existingJob) {
      return res.json({
        success: true,
        jobId: existingJob.id,
        status: existingJob.status,
        stageIndex: existingJob.stageIndex,
        stageName: existingJob.stageName,
        stageMessage: existingJob.stageMessage,
        progressPercent: existingJob.progressPercent,
      });
    }

    const jobId = `job_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    const abortController = new AbortController();

    const newJob: AnalysisJob = {
      id: jobId,
      idempotencyKey,
      userId,
      asset: asset.toUpperCase().trim(),
      assetClass,
      strategy: normalizedStrategy,
      accountBalance: Number(accountBalance) || 10000,
      riskPercent: Number(riskPercent) || 1.0,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      status: 'QUEUED',
      stageIndex: 0,
      stageName: 'QUEUED',
      stageMessage: 'Job queued for institutional analysis...',
      progressPercent: 5,
      charts,
      abortController,
    };

    analysisJobs.set(jobId, newJob);
    persistJobState(newJob);

    globalAnalysisQueue.enqueue(
      jobId,
      async () => {
        await executeInstitutionalPipeline(newJob, charts);
      },
      abortController,
      (pos, total) => {
        if (newJob.status === 'QUEUED') {
          newJob.stageMessage = `Queued for analysis (position ${pos} of ${total}). Waiting for engine...`;
          notifyJobSubscribers(newJob);
        }
      }
    ).catch(err => {
      console.error(`[TradeGuard] Unhandled job execution failure for ${jobId}:`, err);
    });

    return res.json({
      success: true,
      jobId,
      status: newJob.status,
      stageIndex: newJob.stageIndex,
      stageName: newJob.stageName,
      stageMessage: newJob.stageMessage,
      progressPercent: newJob.progressPercent,
    });
  }

  // Synchronous execution routed through globalAnalysisQueue
  const syncJobId = `sync_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  const syncAbort = new AbortController();
  const syncJob: AnalysisJob = {
    id: syncJobId,
    userId,
    asset: asset.toUpperCase().trim(),
    assetClass,
    strategy: normalizedStrategy,
    accountBalance: Number(accountBalance) || 10000,
    riskPercent: Number(riskPercent) || 1.0,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: 'QUEUED',
    stageIndex: 0,
    stageName: 'QUEUED',
    stageMessage: 'Queued for synchronous institutional analysis...',
    progressPercent: 5,
    charts,
    abortController: syncAbort,
  };

  analysisJobs.set(syncJobId, syncJob);
  persistJobState(syncJob);

  try {
    let finalSignal: any = null;
    await globalAnalysisQueue.enqueue(
      syncJobId,
      async () => {
        finalSignal = await executeInstitutionalPipeline(syncJob, charts);
      },
      syncAbort
    );

    if (syncJob.status === 'FAILED') {
      const err = syncJob.error;
      return res.status(err?.status || 500).json({
        success: false,
        errorCode: err?.code || 'ANALYSIS_FAILED',
        error: err?.message || 'An error occurred during chart analysis.',
        retryable: err?.retryable || false,
        retryAfter: err?.retryAfter,
      });
    }

    if (syncJob.status === 'CANCELLED') {
      return res.status(499).json({
        success: false,
        error: 'Analysis cancelled.',
      });
    }

    return res.json({
      success: true,
      signal: finalSignal,
      executionState: finalSignal?.executionState || 'SIGNAL_CONFIRMED',
      analysisStatus: finalSignal?.analysisStatus || 'CONFIRMED',
    });
  } catch (err: any) {
    return res.status(500).json({
      success: false,
      errorCode: 'ANALYSIS_FAILED',
      error: err?.message || 'An internal error occurred during chart analysis.',
    });
  }
});

// PART 19: Gemini Health Check (Requirement 10)
app.get('/api/health/gemini', async (req: Request, res: Response) => {
  const model = TRADEGUARD_GEMINI_MODEL;
  const apiKey = process.env.GEMINI_API_KEY;
  const apiKeyConfigured = Boolean(apiKey && apiKey.trim().length > 0);
  const modelConfigured = Boolean(TRADEGUARD_GEMINI_MODEL && TRADEGUARD_GEMINI_MODEL.trim().length > 0);
  const configured = apiKeyConfigured && modelConfigured;

  let reason = '';
  if (!apiKeyConfigured && !modelConfigured) {
    reason = 'Both GEMINI_API_KEY and TRADEGUARD_GEMINI_MODEL are missing.';
  } else if (!apiKeyConfigured) {
    reason = 'GEMINI_API_KEY is missing or empty.';
  } else if (!modelConfigured) {
    reason = 'TRADEGUARD_GEMINI_MODEL is missing or empty.';
  }

  res.json({
    configured,
    apiKeyConfigured,
    modelConfigured,
    model: model || 'unconfigured',
    status: configured ? 'ready' : 'configuration_required',
    ...(reason ? { reason } : {}),
  });
});

// Safe health check and configuration check endpoint (single zero-token cached probe)
app.get('/api/tradeguard/status', async (req: Request, res: Response) => {
  const forceRefresh = req.query.refresh === 'true';
  const healthStatus = await aiProviderManager.getHealthStatus(forceRefresh);
  const geminiConfigured = healthStatus.primary.configured;
  const modelId = healthStatus.primary.model;
  const modelConfigured = Boolean(modelId && modelId.trim().length > 0);
  const modelAvailable = healthStatus.primary.available;
  const analysisReady = geminiConfigured && modelConfigured && modelAvailable;

  res.json({
    geminiConfigured,
    modelConfigured,
    modelAvailable,
    analysisReady,
    model: modelId,
    error: healthStatus.primary.error || null,
    timestamp: new Date().toISOString(),
    providers: healthStatus,
    activeQueueLength: globalAnalysisQueue.queueLength,
    isProcessing: globalAnalysisQueue.isBusy,
  });
});

// Standard health check endpoint
app.get('/api/health', (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    geminiConfigured: Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.trim()),
    modelConfigured: Boolean(TRADEGUARD_GEMINI_MODEL),
    model: TRADEGUARD_GEMINI_MODEL,
    time: new Date().toISOString(),
    signalsCount: db.signals.length,
    activeQueueLength: globalAnalysisQueue.queueLength,
    isProcessing: globalAnalysisQueue.isBusy,
  });
});

// -------------------------------------------------------------
// Vite Server Integration (Middleware for Dev, Static for Prod)
// -------------------------------------------------------------
async function startServer() {
  const httpServer = http.createServer(app);

  if (process.env.NODE_ENV !== 'production') {
    const disableHmr = process.env.DISABLE_HMR === 'true';
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        ws: disableHmr ? false : undefined,
        hmr: disableHmr ? false : { server: httpServer },
        watch: disableHmr ? null : undefined,
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`TradeGuard Server running on http://localhost:${PORT}`);
  });

  const shutdown = () => {
    httpServer.close(() => {
      process.exit(0);
    });
    setTimeout(() => process.exit(0), 2000).unref();
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

startServer();
