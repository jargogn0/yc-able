const { app, BrowserWindow, shell, Menu } = require('electron');
const path = require('path');

const APP_URL = process.env.LABS_URL || 'https://yc-able.com/app';

let mainWindow;

// Offline/error fallback page shown when network unavailable
const OFFLINE_HTML = `data:text/html,<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>*{margin:0;padding:0;box-sizing:border-box}
body{background:#09090b;color:#fff;font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;text-align:center}
.card{padding:48px;max-width:380px}
.logo{width:48px;height:48px;background:#6366f1;border-radius:12px;display:inline-flex;
align-items:center;justify-content:center;font-size:18px;font-weight:600;margin-bottom:24px}
h2{font-size:20px;font-weight:500;margin-bottom:10px;letter-spacing:-.3px}
p{font-size:13px;color:rgba(255,255,255,.4);line-height:1.6;margin-bottom:28px}
button{background:#6366f1;color:#fff;border:none;padding:10px 24px;border-radius:8px;
font-size:13px;font-weight:500;cursor:pointer;transition:background .15s}
button:hover{background:#5558e8}</style></head>
<body><div class="card">
<div class="logo">19</div>
<h2>Can't connect</h2>
<p>19Labs needs internet to load.<br>Check your connection and try again.</p>
<button onclick="window.location.reload()">Retry</button>
</div></body></html>`;

function destroySplash(splash) {
  try { if (splash && !splash.isDestroyed()) splash.destroy(); } catch (_) {}
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#09090b',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    show: false,
  });

  const splash = new BrowserWindow({
    width: 380,
    height: 280,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
  });
  splash.loadFile(path.join(__dirname, 'splash.html'));
  splash.center();

  // Safety net: always show main window after 8s regardless
  const splashTimeout = setTimeout(() => {
    destroySplash(splash);
    if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.isVisible()) {
      mainWindow.show();
      mainWindow.focus();
    }
  }, 8000);

  mainWindow.loadURL(APP_URL);

  mainWindow.webContents.on('did-finish-load', () => {
    clearTimeout(splashTimeout);
    destroySplash(splash);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.show();
      mainWindow.focus();
    }
  });

  let retryCount = 0;
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    // Ignore aborted loads (e.g. navigating away before page finished)
    if (errorCode === -3) return;
    retryCount++;
    if (retryCount <= 3) {
      // Retry with backoff: 2s, 4s, 8s
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(APP_URL);
      }, Math.pow(2, retryCount) * 1000);
    } else {
      // Give up — show offline page so user isn't stuck on black screen
      clearTimeout(splashTimeout);
      destroySplash(splash);
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.loadURL(OFFLINE_HTML);
        mainWindow.show();
        mainWindow.focus();
      }
    }
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http')) shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    clearTimeout(splashTimeout);
    mainWindow = null;
  });
}

function buildMenu() {
  const isMac = process.platform === 'darwin';
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(isMac ? [{ label: '19Labs', submenu: [
      { role: 'about' },
      { type: 'separator' },
      { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: () => mainWindow && mainWindow.webContents.loadURL(APP_URL) },
      { type: 'separator' },
      { role: 'quit' }
    ]}] : []),
    { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: 'View', submenu: [{ role: 'reload' }, { role: 'toggleDevTools' }, { type: 'separator' }, { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { type: 'separator' }, { role: 'togglefullscreen' }] },
    { label: 'Window', submenu: [{ role: 'minimize' }, { role: 'zoom' }, ...(isMac ? [{ type: 'separator' }, { role: 'front' }] : [{ role: 'close' }])] },
  ]));
}

app.whenReady().then(() => {
  buildMenu();
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
