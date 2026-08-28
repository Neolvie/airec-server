// Сессии и защита логина от перебора для веб-интерфейса.
const crypto = require('crypto');

const SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;
const FAIL_WINDOW_MS = 15 * 60 * 1000;
const MAX_FAILS_PER_IP = 5;
const BLOCK_STEPS_MS = [15 * 60 * 1000, 60 * 60 * 1000, 24 * 60 * 60 * 1000];
const GLOBAL_FAIL_WINDOW_MS = 60 * 60 * 1000;
const GLOBAL_MAX_FAILS = 20;
const GLOBAL_BLOCK_MS = 60 * 60 * 1000;
const FAIL_DELAY_MS = 1000;
const COOKIE_NAME = 'airec_session';

// Сравнение строк без утечки длины и содержимого через время ответа
function hashEqual(a, b) {
  const ha = crypto.createHash('sha256').update(String(a), 'utf8').digest();
  const hb = crypto.createHash('sha256').update(String(b), 'utf8').digest();
  return crypto.timingSafeEqual(ha, hb);
}

function parseCookies(header) {
  const out = {};
  for (const part of String(header || '').split(';')) {
    const i = part.indexOf('=');
    if (i > 0) out[part.slice(0, i).trim()] = part.slice(i + 1).trim();
  }
  return out;
}

class Auth {
  constructor({ login, password, log }) {
    this.login = login;
    this.password = password;
    this.log = log || (() => {});
    this.sessions = new Map(); // token -> expiresAt
    this.ipFails = new Map(); // ip -> { times, level, blockedUntil }
    this.globalFails = [];
    this.globalBlockedUntil = 0;
  }

  get enabled() {
    return Boolean(this.login && this.password);
  }

  hasSession(req) {
    const token = parseCookies(req.headers.cookie)[COOKIE_NAME];
    if (!token) return false;
    const expires = this.sessions.get(token);
    if (!expires) return false;
    if (Date.now() > expires) {
      this.sessions.delete(token);
      return false;
    }
    return true;
  }

  require() {
    return (req, res, next) => {
      if (!this.enabled) return res.status(503).json({ error: 'web ui disabled' });
      if (!this.hasSession(req)) return res.status(401).json({ error: 'unauthorized' });
      return next();
    };
  }

  blockedForMs(ip) {
    const now = Date.now();
    if (now < this.globalBlockedUntil) return this.globalBlockedUntil - now;
    const rec = this.ipFails.get(ip);
    if (rec && now < rec.blockedUntil) return rec.blockedUntil - now;
    return 0;
  }

  registerFail(ip) {
    const now = Date.now();
    this.globalFails = this.globalFails.filter((t) => now - t < GLOBAL_FAIL_WINDOW_MS);
    this.globalFails.push(now);
    if (this.globalFails.length >= GLOBAL_MAX_FAILS) {
      this.globalBlockedUntil = now + GLOBAL_BLOCK_MS;
      this.log(`LOGIN: глобальная блокировка на ${GLOBAL_BLOCK_MS / 60000} мин ` +
        `(${this.globalFails.length} неудач за час со всех IP)`);
    }
    const rec = this.ipFails.get(ip) || { times: [], level: 0, blockedUntil: 0 };
    rec.times = rec.times.filter((t) => now - t < FAIL_WINDOW_MS);
    rec.times.push(now);
    if (rec.times.length >= MAX_FAILS_PER_IP) {
      const step = BLOCK_STEPS_MS[Math.min(rec.level, BLOCK_STEPS_MS.length - 1)];
      rec.blockedUntil = now + step;
      rec.level += 1;
      rec.times = [];
      this.log(`LOGIN: блокировка ${ip} на ${step / 60000} мин`);
    }
    this.ipFails.set(ip, rec);
  }

  handleLogin() {
    return (req, res) => {
      if (!this.enabled) return res.status(503).json({ error: 'web ui disabled' });
      const ip = req.ip;
      const blockedMs = this.blockedForMs(ip);
      if (blockedMs > 0) {
        const minutes = Math.ceil(blockedMs / 60000);
        return res.status(429).json({ error: `Слишком много попыток. Подождите ${minutes} мин.` });
      }
      const { login, password } = req.body || {};
      const loginOk = hashEqual(login || '', this.login);
      const passwordOk = hashEqual(password || '', this.password);
      if (!loginOk || !passwordOk) {
        this.registerFail(ip);
        this.log(`LOGIN: неудачная попытка с ${ip}`);
        return setTimeout(
          () => res.status(401).json({ error: 'Неверный логин или пароль' }), FAIL_DELAY_MS);
      }
      this.ipFails.delete(ip);
      this.pruneSessions();
      const token = crypto.randomBytes(32).toString('hex');
      this.sessions.set(token, Date.now() + SESSION_TTL_MS);
      const secure = req.secure || req.headers['x-forwarded-proto'] === 'https';
      res.setHeader('Set-Cookie',
        `${COOKIE_NAME}=${token}; Max-Age=${SESSION_TTL_MS / 1000}; Path=/; HttpOnly; ` +
        `SameSite=Lax${secure ? '; Secure' : ''}`);
      this.log(`LOGIN: успешный вход с ${ip}`);
      return res.json({ ok: true });
    };
  }

  handleLogout() {
    return (req, res) => {
      const token = parseCookies(req.headers.cookie)[COOKIE_NAME];
      if (token) this.sessions.delete(token);
      res.setHeader('Set-Cookie',
        `${COOKIE_NAME}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax`);
      res.json({ ok: true });
    };
  }

  pruneSessions() {
    const now = Date.now();
    for (const [token, expires] of this.sessions) {
      if (now > expires) this.sessions.delete(token);
    }
  }
}

module.exports = { Auth };
