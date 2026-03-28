const { app, BrowserWindow, shell, Menu } = require('electron');
const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

app.disableHardwareAcceleration();

// Root yc-able.com has no A record — www CNAME → elqdcfzi.up.railway.app
const BACKEND = 'www.yc-able.com';
let localPort = 0;
let mainWindow;

function proxyRequest(req, res) {
  const options = {
    hostname: BACKEND,
    port: 443,
    path: req.url,
    method: req.method,
    headers: { ...req.headers, host: BACKEND, origin: `https://${BACKEND}`, referer: `https://${BACKEND}/` },
  };
  delete options.headers['accept-encoding'];

  const proxy = https.request(options, (backRes) => {
    res.writeHead(backRes.statusCode, {
      ...backRes.headers,
      'access-control-allow-origin': '*',
      'access-control-allow-credentials': 'true',
    });
    backRes.pipe(res, { end: true });
  });
  proxy.on('error', (e) => {
    console.error('[proxy]', e.message);
    res.writeHead(502, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: e.message }));
  });
  req.pipe(proxy, { end: true });
}

function startServer(callback) {
  const server = http.createServer((req, res) => {
    const url = req.url || '/';
    const isProxy = url.startsWith('/api/') || url.startsWith('/auth/') ||
                    url.startsWith('/download/') || url.startsWith('/ws');
    if (isProxy) { proxyRequest(req, res); return; }

    const file = url === '/' ? 'app.html' : url.replace(/^\//, '');
    const filePath = path.join(__dirname, file);
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); res.end('Not found'); return; }
      const ext = path.extname(filePath).slice(1).toLowerCase();
      const types = { html: 'text/html', js: 'text/javascript', css: 'text/css', png: 'image/png' };
      let content = data;
      if (ext === 'html') {
        content = data.toString('utf-8')
          .replace(/content="https:\/\/(?:www\.)?yc-able\.com"/g, `content="http://127.0.0.1:${localPort}"`);
      }
      res.writeHead(200, { 'Content-Type': (types[ext] || 'text/plain') + '; charset=utf-8' });
      res.end(content);
    });
  });

  server.listen(0, '127.0.0.1', () => {
    localPort = server.address().port;
    console.log(`[server] http://127.0.0.1:${localPort} → proxying to https://${BACKEND}`);
    callback(localPort);
  });
}

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1440, height: 900, minWidth: 900, minHeight: 600,
    titleBarStyle: 'hiddenInset', trafficLightPosition: { x: 16, y: 17 },
    backgroundColor: '#09090b', show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true, nodeIntegration: false,
      webSecurity: false, allowRunningInsecureContent: true, backgroundThrottling: false,
    },
  });

  mainWindow.loadURL(`http://127.0.0.1:${port}/app.html`);
  mainWindow.once('ready-to-show', () => { mainWindow.show(); });
  setTimeout(() => {
    if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.isVisible()) mainWindow.show();
  }, 10000);

  mainWindow.webContents.on('console-message', (e, level, msg, line, src) => {
    console.log(`[renderer:${level}] ${msg} (${src}:${line})`);
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http')) shell.openExternal(url);
    return { action: 'deny' };
  });
  mainWindow.on('closed', () => { mainWindow = null; });
}

function buildMenu() {
  const isMac = process.platform === 'darwin';
  const reload = () => { if (mainWindow && !mainWindow.isDestroyed()) mainWindow.reload(); };
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(isMac ? [{ label: '19Labs', submenu: [
      { role: 'about' }, { type: 'separator' },
      { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: reload },
      { type: 'separator' }, { role: 'services' }, { type: 'separator' },
      { role: 'hide' }, { role: 'hideOthers' }, { role: 'unhide' },
      { type: 'separator' }, { role: 'quit' }
    ]}] : []),
    { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: 'View', submenu: [
      { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: reload },
      { role: 'toggleDevTools' }, { type: 'separator' },
      { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
      { type: 'separator' }, { role: 'togglefullscreen' }
    ]},
    { label: 'Window', submenu: [{ role: 'minimize' }, { role: 'zoom' },
      ...(isMac ? [{ type: 'separator' }, { role: 'front' }] : [{ role: 'close' }])] },
  ]));
}

app.whenReady().then(() => {
  buildMenu();
  startServer((port) => { createWindow(port); });
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(localPort); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
