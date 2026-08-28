const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const PORT = process.env.PORT || 8080;
const UPLOAD_DIR = process.env.UPLOAD_DIR || path.join(__dirname, 'uploads');
const TMP_DIR = path.join(UPLOAD_DIR, '.tmp');

fs.mkdirSync(TMP_DIR, { recursive: true });

const app = express();
const upload = multer({
  dest: TMP_DIR,
  limits: { fileSize: 500 * 1024 * 1024 }
});

// Приводит sn/fileName к безопасному имени файла (без путей и спецсимволов)
function sanitize(name) {
  return String(name || '').replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 200);
}

function log(...args) {
  console.log(new Date().toISOString(), ...args);
}

app.post('/api/upload', upload.single('file'), (req, res) => {
  const cleanup = () => {
    if (req.file) fs.unlink(req.file.path, () => {});
  };

  try {
    const { fileName, sn } = req.body || {};

    if (!req.file) {
      log('REJECT: no file field', JSON.stringify(req.body));
      cleanup();
      return res.status(400).send('missing file');
    }
    if (!fileName || !sn) {
      log('REJECT: missing fileName or sn', JSON.stringify(req.body));
      cleanup();
      return res.status(400).send('missing fileName or sn');
    }

    const safeSn = sanitize(sn);
    const safeName = sanitize(fileName);
    const deviceDir = path.join(UPLOAD_DIR, safeSn);
    const dest = path.join(deviceDir, safeName);

    if (fs.existsSync(dest)) {
      log(`DUPLICATE: ${safeSn}/${safeName} (${req.file.size} bytes) — already stored, ok`);
      cleanup();
      return res.status(200).send('success');
    }

    fs.mkdirSync(deviceDir, { recursive: true });
    fs.renameSync(req.file.path, dest);
    log(`SAVED: ${safeSn}/${safeName} (${req.file.size} bytes)`);
    return res.status(200).send('success');
  } catch (err) {
    log('ERROR:', err.message);
    cleanup();
    return res.status(500).send('error');
  }
});

// Список принятых файлов — для проверки, что загрузки доходят
app.get('/api/files', (req, res) => {
  const result = [];
  let devices = [];
  try {
    devices = fs.readdirSync(UPLOAD_DIR).filter((d) => !d.startsWith('.'));
  } catch {}
  for (const sn of devices) {
    const dir = path.join(UPLOAD_DIR, sn);
    if (!fs.statSync(dir).isDirectory()) continue;
    for (const f of fs.readdirSync(dir)) {
      const st = fs.statSync(path.join(dir, f));
      result.push({ sn, fileName: f, size: st.size, uploadedAt: st.mtime.toISOString() });
    }
  }
  result.sort((a, b) => b.uploadedAt.localeCompare(a.uploadedAt));
  res.json(result);
});

app.get('/', (req, res) => {
  res.send('AIREC upload server is running. POST /api/upload, GET /api/files');
});

app.use((err, req, res, next) => {
  log('ERROR:', err.message);
  res.status(500).send('error');
});

app.listen(PORT, () => {
  log(`AIREC upload server listening on port ${PORT}, storing files in ${UPLOAD_DIR}`);
});
