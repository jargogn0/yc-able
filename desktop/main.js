const { app, BrowserWindow, shell, Menu } = require('electron');
const path = require('path');

app.disableHardwareAcceleration();

const APP_URL = process.env.LABS_URL || 'https://yc-able.com/app';
const DEBUG = process.env.LABS_DEBUG === '1';

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
    show: true,   // show immediately
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,
      allowRunningInsecureContent: true,
      backgroundThrottling: false,
    },
  });

  // Splash overlay
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

  let splashClosed = false;
  function closeSplash() {
    if (splashClosed) return;
    splashClosed = true;
    try { if (!splash.isDestroyed()) splash.destroy(); } catch (_) {}
  }

  mainWindow.loadURL(APP_URL);

  // Log all console messages from the renderer for debugging
  mainWindow.webContents.on('console-message', (e, level, msg, line, src) => {
    if (level >= 2) console.error(`[renderer] ${msg} (${src}:${line})`);
  });

  // did-finish-load fires when HTML is parsed; give JS 3s to run and paint
  mainWindow.webContents.on('did-finish-load', () => {
    setTimeout(closeSplash, 3000);
  });

  // Safety net — close splash after 15s no matter what
  setTimeout(closeSplash, 15000);

  // Retry on network failure
  let retries = 0;
  mainWindow.webContents.on('did-fail-load', (e, code) => {
    if (code === -3) return;
    retries++;
    if (retries <= 3) {
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(APP_URL);
      }, retries * 2000);
    } else {
      closeSplash();
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
      { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: () => {
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
