const { app, BrowserWindow, shell, Menu } = require('electron');
const path = require('path');

const APP_URL = process.env.LABS_URL || 'https://yc-able.com/app';

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#09090b',
    show: true,   // show immediately — no more black screen waiting
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,              // allow mixed content / remote resources
      allowRunningInsecureContent: true,
      partition: 'persist:19labs',     // persistent session — keeps login cookies
    },
  });

  // Show splash as an overlay on top of the main window
  const splash = new BrowserWindow({
    width: 380,
    height: 280,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    parent: mainWindow,
  });
  splash.loadFile(path.join(__dirname, 'splash.html'));
  splash.center();

  mainWindow.loadURL(APP_URL);

  function closeSplash() {
    try { if (splash && !splash.isDestroyed()) splash.destroy(); } catch (_) {}
  }

  // Close splash when page finishes loading
  mainWindow.webContents.on('did-finish-load', () => {
    closeSplash();
  });

  // On load failure: retry up to 3x then close splash so user isn't stuck
  let retries = 0;
  mainWindow.webContents.on('did-fail-load', (e, code) => {
    if (code === -3) return; // aborted (navigation)
    retries++;
    if (retries <= 3) {
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(APP_URL);
      }, retries * 2000);
    } else {
      closeSplash();
      mainWindow.loadURL(`data:text/html,<html><body style="background:#09090b;color:#fff;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center"><div><div style="width:48px;height:48px;background:#6366f1;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:600;margin:0 auto 20px">19</div><h2 style="margin:0 0 8px;font-weight:500">Can't connect</h2><p style="color:rgba(255,255,255,.4);margin:0 0 24px;font-size:13px">Check your internet connection</p><button onclick="location.href='${APP_URL}'" style="background:#6366f1;color:#fff;border:none;padding:10px 24px;border-radius:8px;font-size:13px;cursor:pointer">Retry</button></div></body></html>`);
    }
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http')) shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => { mainWindow = null; });
}

function buildMenu() {
  const isMac = process.platform === 'darwin';
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(isMac ? [{ label: '19Labs', submenu: [
      { role: 'about' },
      { type: 'separator' },
      { label: 'Reload App', accelerator: 'CmdOrCtrl+R', click: () => {
        if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(APP_URL);
      }},
      { type: 'separator' },
      { role: 'services' },
      { type: 'separator' },
      { role: 'hide' }, { role: 'hideOthers' }, { role: 'unhide' },
      { type: 'separator' },
      { role: 'quit' }
    ]}] : []),
    { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: 'View', submenu: [
      { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: () => mainWindow && mainWindow.loadURL(APP_URL) },
      { role: 'toggleDevTools' },
      { type: 'separator' },
      { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
      { type: 'separator' }, { role: 'togglefullscreen' }
    ]},
    { label: 'Window', submenu: [{ role: 'minimize' }, { role: 'zoom' }, ...(isMac ? [{ type: 'separator' }, { role: 'front' }] : [{ role: 'close' }])] },
  ]));
}

app.whenReady().then(() => {
  buildMenu();
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
